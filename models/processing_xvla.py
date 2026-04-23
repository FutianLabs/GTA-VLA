
from __future__ import annotations

import logging
from transformers import ProcessorMixin, AutoTokenizer, AutoImageProcessor, BartTokenizer, BartTokenizerFast
from typing import List, Union, Dict, Any, Optional, Literal
import torch
import numpy as np

logger = logging.getLogger(__name__)


class XVLAProcessor(ProcessorMixin):
    """
    XVLAProcessor: Unified multimodal processor for XVLA models.

    Handles:
      - Multi-view image inputs (e.g., from multiple cameras).
      - Batch processing for multiple samples.
      - Joint tokenization and image tensor preparation.
      - Support for Florence2 and Qwen3-VL backends.

    Attributes
    ----------
    num_views : int, default=3
        Expected number of image views per sample. Missing views will be padded with zeros.
    language_max_length : int, default=128 for Qwen models, 50 for Florence2
        Maximum token length for text encoding (before template expansion for Qwen).
    vlm_backbone_type : str, default="florence2"
        Type of VLM backbone: "florence2" or "qwen3_vl"
    """

    num_views: int = 3
    language_max_length: int = None  # Auto-determined based on backbone type
    vlm_backbone_type: str = "florence2"

    # Hugging Face ProcessorMixin-required metadata
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast", "Qwen2Tokenizer", "Qwen2TokenizerFast")
    def __init__(
        self, 
        image_processor=None, 
        tokenizer=None,
        vlm_backbone_type: str = "florence2",
    ):
        """
        Initialize XVLAProcessor.

        Parameters
        ----------
        image_processor : PreTrainedImageProcessor, optional
            The image processor used to normalize/resize images.
        tokenizer : PreTrainedTokenizer, optional
            The tokenizer used for text tokenization.
        vlm_backbone_type : str, default="florence2"
            Type of VLM backbone to use for processing.
        """
        super().__init__(image_processor, tokenizer)
        self.vlm_backbone_type = vlm_backbone_type
        
        # Auto-set language_max_length if not already set
        if self.language_max_length is None:
            if vlm_backbone_type == "qwen3_vl":
                self.language_max_length = 256
            else:
                self.language_max_length = 50
        
        # For Qwen3-VL, we may have a wrapped processor
        self._qwen_processor = None

    @classmethod
    def from_pretrained_vlm(
        cls,
        vlm_backbone_type: Literal["florence2", "qwen3_vl"],
        pretrained_name_or_path: str,
        num_views: int = 3,
        language_max_length: int = None,
        **kwargs,
    ) -> "XVLAProcessor":
        """
        Create processor from a pretrained VLM model.
        """
        if vlm_backbone_type == "florence2":
            image_processor = AutoImageProcessor.from_pretrained(pretrained_name_or_path, **kwargs)
            try:
                tokenizer = BartTokenizerFast.from_pretrained(pretrained_name_or_path, **kwargs)
            except Exception:
                tokenizer = BartTokenizer.from_pretrained(pretrained_name_or_path, **kwargs)
            processor = cls(
                image_processor=image_processor,
                tokenizer=tokenizer,
                vlm_backbone_type="florence2",
            )
        elif vlm_backbone_type == "qwen3_vl":
            # Qwen3-VL has its own processor (requires transformers >= 4.49.0)
            from transformers import AutoProcessor as Qwen3AutoProcessor
            
            # Pop custom kwargs before passing to HF's from_pretrained
            use_cot = kwargs.pop("use_cot_training", False)
            cot_max_length = kwargs.pop("cot_max_length", 768)
            
            qwen_processor = Qwen3AutoProcessor.from_pretrained(pretrained_name_or_path, **kwargs)
            
            # Add CoT special tokens if needed
            if use_cot:
                additional_tokens = [
                    "<|cot_start|>", "<|cot_end|>",
                    "<|objects_start|>", "<|objects_end|>",
                    "<|pick_start|>", "<|pick_end|>",
                    "<|place_start|>", "<|place_end|>",
                    "<|affordance_2d_start|>", "<|affordance_2d_end|>",
                    "<|gripper_path_2d_start|>", "<|gripper_path_2d_end|>",
                ]
                existing = list(qwen_processor.tokenizer.additional_special_tokens)
                new_tokens = [t for t in additional_tokens if t not in existing]
                if new_tokens:
                    special_tokens = {"additional_special_tokens": existing + new_tokens}
                    num_added_tokens = qwen_processor.tokenizer.add_special_tokens(special_tokens)
                    # No logging/printing here; keep processor quiet.
            
            processor = cls(
                image_processor=qwen_processor.image_processor,
                tokenizer=qwen_processor.tokenizer,
                vlm_backbone_type="qwen3_vl",
            )
            processor._qwen_processor = qwen_processor
            if use_cot:
                processor.cot_max_length = cot_max_length


        processor.num_views = num_views
        
        # Auto-determine language_max_length based on backbone type if not specified
        if language_max_length is None:
            if vlm_backbone_type == "qwen3_vl":
                language_max_length = 196
            else:
                # Florence2: original default
                language_max_length = 50
        
        processor.language_max_length = language_max_length
        return processor

    # ================== LANGUAGE ENCODING ==================
    def encode_language(
        self,
        language_instruction: Union[str, List[str]],
        cot_texts: Optional[List[str]] = None,
        image_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize instructions with pre-expanded image placeholders.

        For Qwen3-VL, ``image_grid_thw`` is required so that the correct number
        of ``<|image_pad|>`` tokens can be inserted (one per visual token after
        patch merging).  This is essential for M-RoPE position computation.

        When *cot_texts* is provided (Qwen3-VL only), instruction + CoT are
        tokenized as a single sequence and a ``labels`` tensor is returned.

        Returns
        -------
        Dict[str, torch.Tensor]
            Always: ``{"input_ids": [B, L]}``
            With CoT: additionally ``{"labels": [B, L]}``
        """
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]

        if self.vlm_backbone_type == "qwen3_vl":
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required for Qwen3-VL encode_language")

            merge_size = getattr(self.image_processor, "merge_size", 2)
            merge_length = merge_size ** 2
            image_pad = "<|image_pad|>"

            formatted_instructions = []
            for idx, inst in enumerate(language_instruction):
                # Only view 0 goes into the VLM text stream
                n = int(image_grid_thw[idx, 0].prod().item()) // merge_length
                image_placeholder = f"<|vision_start|>{image_pad * n}<|vision_end|>"
                formatted = f"<|im_start|>user\n{image_placeholder}{inst}<|im_end|>"
                formatted_instructions.append(formatted)

            if cot_texts is not None:
                cot_max_length = getattr(self, 'cot_max_length', 376)

                combined = []
                for inst_str, cot in zip(formatted_instructions, cot_texts):
                    combined.append(inst_str + cot if cot.strip() else inst_str)

                combined_enc = self.tokenizer(
                    combined,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                )
                input_ids = combined_enc["input_ids"]

                cot_start_id = self.tokenizer.convert_tokens_to_ids("<|cot_start|>")
                labels = input_ids.clone()
                pad_id = self.tokenizer.pad_token_id

                for i in range(input_ids.size(0)):
                    if not cot_texts[i].strip():
                        labels[i] = -100
                        continue
                    cot_positions = (input_ids[i] == cot_start_id).nonzero(as_tuple=True)[0]
                    if len(cot_positions) > 0:
                        labels[i, :cot_positions[0].item()] = -100
                    else:
                        labels[i] = -100
                labels[labels == pad_id] = -100

                return {"input_ids": input_ids, "labels": labels}
            else:
                inputs = self.tokenizer(
                    formatted_instructions,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                )
                return {"input_ids": inputs["input_ids"]}

        else:
            # Florence2: simple tokenization
            inputs = self.tokenizer(
                language_instruction,
                return_tensors="pt",
                padding="max_length",
                max_length=self.language_max_length,
                truncation=True,
            )
            return {"input_ids": inputs["input_ids"]}

    # ================== IMAGE ENCODING ==================
    def encode_image(
        self,
        images: Union[List, List[List]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess one or more sets of multi-view images.

        Parameters
        ----------
        images : List or List[List]
            Single sample: [img1, img2, ...]
            Batch: [[img1a, img1b], [img2a, img2b, img2c], ...]
            Each image may be a PIL.Image, NumPy array, or torch.Tensor.

        Returns
        -------
        Dict[str, torch.Tensor]
            {
              "image_input": tensor [B, num_views, C, H, W],
              "image_mask": tensor [B, num_views],
              "image_grid_thw": tensor [B, V, 3] (only for Qwen3-VL)
            }
        """
        # Normalize to batch form
        if not isinstance(images[0], (list, tuple)):
            images = [images]  # convert single sample to batch of size 1
        
        # Truncate to num_views if more images provided
        images = [sample[:self.num_views] for sample in images]

        if self.vlm_backbone_type == "qwen3_vl":
            return self._encode_image_qwen3(images, **kwargs)
        else:
            return self._encode_image_florence(images, **kwargs)

    def _encode_image_florence(
        self,
        images: List[List],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Florence2-style image encoding."""
        batch_imgs, batch_masks = [], []

        for sample_imgs in images:
            processed = self.image_processor(sample_imgs, return_tensors="pt", **kwargs)["pixel_values"]
            V_exist = processed.size(0)

            # Pad to self.num_views
            if V_exist < self.num_views:
                processed = torch.cat(
                    [processed,
                     processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                    dim=0,
                )

            # Mask: True for valid slots, False for padding
            image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=processed.device)
            image_mask[:V_exist] = True

            batch_imgs.append(processed)
            batch_masks.append(image_mask)

        image_input = torch.stack(batch_imgs, dim=0)  # [B, num_views, C, H, W]
        image_mask = torch.stack(batch_masks, dim=0)  # [B, num_views]

        return {"image_input": image_input, "image_mask": image_mask}

    def _encode_image_qwen3(
        self,
        images: List[List],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Qwen3-VL style image encoding using NATIVE processor.
        
        Returns pre-embedded patches [num_patches, embed_dim] from Qwen3-VL's
        pre-trained Conv3d patch embedding, preserving full model capacity.
        """
        from PIL import Image as PILImage
        
        batch_imgs, batch_masks = [], []
        all_grid_thw = []

        for sample_imgs in images:
            sample_processed = []
            sample_grid_thw = []
            
            for img in sample_imgs:
                # Convert to PIL Image if needed
                if not isinstance(img, PILImage.Image):
                    if isinstance(img, np.ndarray):
                        img = PILImage.fromarray(img)
                    else:
                        raise ValueError(f"Unsupported image type: {type(img)}")
                
                vlm_kwargs = kwargs.copy()
                vlm_kwargs.pop("return_tensors", None)
                processed = self.image_processor([img], return_tensors="pt", **vlm_kwargs)
                pixel_values = processed["pixel_values"]  # [num_patches, 1536]
                
                if "image_grid_thw" in processed:
                    sample_grid_thw.append(processed["image_grid_thw"])
                else:
                    import logging, math
                    logging.warning("image_grid_thw not in processor output, computing fallback")
                    num_patches = pixel_values.shape[0]
                    grid_size = int(math.sqrt(num_patches))
                    sample_grid_thw.append(torch.tensor([[1, grid_size, grid_size]]))
                
                sample_processed.append(pixel_values)
            
            V_exist = len(sample_processed)
            
            if sample_processed:
                max_patches = max(p.shape[0] for p in sample_processed)
                embed_dim = sample_processed[0].shape[1]
                
                padded_views = []
                for patches in sample_processed:
                    if patches.shape[0] < max_patches:
                        pad = torch.zeros(max_patches - patches.shape[0], embed_dim, 
                                         device=patches.device, dtype=patches.dtype)
                        patches = torch.cat([patches, pad], dim=0)
                    padded_views.append(patches)
                
                stacked = torch.stack(padded_views, dim=0)  # [V_exist, max_patches, embed_dim]
                
                if V_exist < self.num_views:
                    pad_views = torch.zeros(self.num_views - V_exist, max_patches, embed_dim,
                                           device=stacked.device, dtype=stacked.dtype)
                    stacked = torch.cat([stacked, pad_views], dim=0)
                    if sample_grid_thw:
                        pad_grid = torch.zeros(self.num_views - V_exist, 3, dtype=torch.long)
                        sample_grid_thw.append(pad_grid)
            else:
                # Fallback: use embed_dim from processor config if available
                embed_dim = getattr(self.image_processor, 'hidden_size', 1536)
                stacked = torch.zeros(self.num_views, 256, embed_dim)
                sample_grid_thw = [torch.tensor([[1, 16, 16]])] * self.num_views
            
            batch_imgs.append(stacked)
            
            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            image_mask[:V_exist] = True
            batch_masks.append(image_mask)
            
            if sample_grid_thw:
                all_grid_thw.append(torch.cat(sample_grid_thw, dim=0))

        # Batch-level padding for different max_patches across samples
        if batch_imgs:
            batch_max_patches = max(x.shape[1] for x in batch_imgs)
            if any(x.shape[1] != batch_max_patches for x in batch_imgs):
                padded_batch_imgs = []
                for x in batch_imgs:
                    if x.shape[1] < batch_max_patches:
                        pad = x.new_zeros((x.shape[0], batch_max_patches - x.shape[1], x.shape[2]))
                        x = torch.cat([x, pad], dim=1)
                    padded_batch_imgs.append(x)
                batch_imgs = padded_batch_imgs

        image_input = torch.stack(batch_imgs, dim=0)  # [B, num_views, max_patches, embed_dim]
        image_mask = torch.stack(batch_masks, dim=0)   # [B, num_views]
        
        result = {"image_input": image_input, "image_mask": image_mask}

        if all_grid_thw:
            result["image_grid_thw"] = torch.stack(all_grid_thw, dim=0)  # [B, V, 3]
        
        return result

    # ================== COMBINED CALL ==================
    def __call__(
        self,
        images: Optional[Union[List, List[List]]] = None,
        language_instruction: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Combine image and text encoding into a unified multimodal input.

        Images are encoded **first** so that ``image_grid_thw`` is available
        for ``encode_language`` (needed by Qwen3-VL to pre-expand placeholders).
        """
        outputs: Dict[str, Any] = {}

        # Encode image FIRST to get image_mask / image_grid_thw
        if images is not None:
            outputs.update(self.encode_image(images, **kwargs))

        # Encode language with image metadata
        if language_instruction is not None:
            outputs.update(self.encode_language(
                language_instruction,
                image_mask=outputs.get("image_mask"),
                image_grid_thw=outputs.get("image_grid_thw"),
            ))

        if "input_ids" in outputs and "image_input" in outputs:
            assert outputs["input_ids"].size(0) == outputs["image_input"].size(0), (
                f"Batch mismatch: text batch {outputs['input_ids'].size(0)} "
                f"!= image batch {outputs['image_input'].size(0)}"
            )
        return outputs


# =============================================================================
# Factory function for creating processor
# =============================================================================

def build_xvla_processor(
    vlm_backbone_type: Literal["florence2", "qwen3_vl"] = "florence2",
    pretrained_name_or_path: Optional[str] = None,
    num_views: int = 3,
    language_max_length: int = None,
    use_cot_training: bool = False,
    cot_max_length: int = 768,
    **kwargs,
) -> XVLAProcessor:
    """Factory function to build XVLA processor based on VLM backbone type."""
    if pretrained_name_or_path is None:
        if vlm_backbone_type == "qwen3_vl":
            pretrained_name_or_path = "Qwen/Qwen3-VL-2B-Instruct"
        else:
            pretrained_name_or_path = "microsoft/Florence-2-large"
    
    return XVLAProcessor.from_pretrained_vlm(
        vlm_backbone_type=vlm_backbone_type,
        pretrained_name_or_path=pretrained_name_or_path,
        num_views=num_views,
        language_max_length=language_max_length,
        use_cot_training=use_cot_training,
        cot_max_length=cot_max_length,
        **kwargs,
    )


class GTAVLAProcessor(XVLAProcessor):
    """Primary public processor name for GTA-VLA."""


def build_gtavla_processor(
    vlm_backbone_type: Literal["florence2", "qwen3_vl"] = "florence2",
    pretrained_name_or_path: Optional[str] = None,
    num_views: int = 3,
    language_max_length: int = None,
    use_cot_training: bool = False,
    cot_max_length: int = 768,
    **kwargs,
) -> GTAVLAProcessor:
    if pretrained_name_or_path is None:
        if vlm_backbone_type == "qwen3_vl":
            pretrained_name_or_path = "Qwen/Qwen3-VL-2B-Instruct"
        else:
            pretrained_name_or_path = "microsoft/Florence-2-large"

    return GTAVLAProcessor.from_pretrained_vlm(
        vlm_backbone_type=vlm_backbone_type,
        pretrained_name_or_path=pretrained_name_or_path,
        num_views=num_views,
        language_max_length=language_max_length,
        use_cot_training=use_cot_training,
        cot_max_length=cot_max_length,
        **kwargs,
    )
