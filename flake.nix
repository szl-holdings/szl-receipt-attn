{
  description = "Flake for szl-receipt-attn (torch-noarch Triton JIT)";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
      # Test-shell only. Not shipped in the Hub artifact. No benches.
      pythonCheckInputs = pkgs: with pkgs; [ pytest ];
    };
}
