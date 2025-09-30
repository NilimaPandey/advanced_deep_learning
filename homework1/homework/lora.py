class LoRALinear(HalfLinear):
    def __init__(self, in_features: int, out_features: int, lora_dim: int = 32, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.requires_grad_(False)

        self.lora_A = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_B = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Init to minimize deviation
        torch.nn.init.normal_(self.lora_A.weight, std=1e-4)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(x)
        lora_out = self.lora_B(self.lora_A(x.to(torch.float32)))
        return base_out + lora_out.to(x.dtype)
