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
      # pytest is for testshell only. Do not add it to pythonBuildInputs —
      # that would ship pytest as a kernel runtime dependency.
      pythonCheckInputs = pkgs: with pkgs; [ pytest ];
    };
}
