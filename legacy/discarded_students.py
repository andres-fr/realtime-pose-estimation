class RefinerStudent(torch.nn.Module):
    """
    """

    REMARKS = "First attempt. This arch lacks the context capabilities. Also gradients dont propagate
    well through it."

    def __init__(self, hhrnet_statedict_path=None, device="cuda",
                 layers_per_stage=[3, 3, 3],
                 num_heatmaps=17, ae_dims=1,
                 half_precision=True, init_fn=torch.nn.init.kaiming_normal_,
                 trainable_stem=False, bn_momentum=0.1):
        """
        """
        super().__init__()
        self.bn_momentum = bn_momentum
        self.layers_per_stage = layers_per_stage
        self.num_heatmaps = num_heatmaps
        self.ae_dims = ae_dims
        # load stem
        self.stem = StemHRNet()
        self.stem_out_chans = self.stem.layer1[-1].bn3.num_features
        self.trainable_stem = trainable_stem
        #
        # load body
        self.stages = self._make_body()
        # optionally initialize parameters
        if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, 0.0))
        if half_precision:
            self.stem = network_to_half(self.stem)
        else:
            self.stem = torch.nn.Sequential(torch.nn.Identity(), self.stem)
        if hhrnet_statedict_path is not None:
            self.stem[1].load_pretrained(
                hhrnet_statedict_path, device, check=False)
        #
        self.to(device)

    def save_body(self, out_path):
        """
        """
        torch.save(self.stages.state_dict(), out_path)

    def load_body(self, statedict_path):
        """
        """
        self.stages.load_state_dict(torch.load(statedict_path))

    def _make_body(self):
        """
        """
        stages = torch.nn.ModuleList()
        ch = self.stem_out_chans
        # add all stages but last with same number of channels
        for l in self.layers_per_stage[:-1]:
            in_chans = [ch for _ in range(l)]
            out_chans = [ch for _ in range(l)]
            stage = get_straight_skip_conv(in_chans, out_chans,
                                           self.bn_momentum)
            stages.append(stage)
        # last stage outputs num_heatmaps + ae_dims
        last_l = self.layers_per_stage[-1]
        in_chans = [ch for _ in range(last_l)]
        out_chans = [ch for _ in range(last_l)]
        out_chans[-1] = self.num_heatmaps + self.ae_dims
        stage = get_straight_skip_conv(in_chans, out_chans,
                                       self.bn_momentum)
        stages.append(stage)
        #
        return stages  # torch.nn.Sequential(*stages)

    def forward(self, x, out_hw=None):
        """
        """
        if self.trainable_stem:
            stem_out = self.stem(x)
        else:
            with torch.no_grad():
                stem_out = self.stem(x)
        # progressive context refinement
        x = self.stages[0](stem_out)
        for s in self.stages[1:]:
            x = s(stem_out + x)
        # resize last stage
        if out_hw is not None:
            x = torch.nn.functional.interpolate(
                x, out_hw, mode="bilinear", align_corners=True)
        return x
