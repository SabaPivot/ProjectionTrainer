import torch
import torch.nn as nn

class MLPProjector(nn.Module):
    """
    A simple 2-layer MLP to project vision features to the LLM embedding space.
    Inspired by CheXagent paper's description, allowing variable expansion.
    Projects each patch embedding individually.
    """
    # TODO: Experiment with the intermediate dimensions of the MLP projector.
    # The CheXagent paper used a larger expansion (e.g., 10x input dim).
    # Consider adding an argument to control the intermediate dimension size/ratio.
    def __init__(self, vision_dim: int, llm_dim: int, expansion_factor: int = 1):
        super().__init__()
        intermediate_dim = vision_dim * expansion_factor
        self.model = nn.Sequential(
            nn.Linear(vision_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, llm_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, num_patches, vision_dim]
        Returns:
            Tensor of shape [batch_size, num_patches, llm_dim]
        """
        return self.model(x)
