class baseline_resnet50_init_reuse(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(baseline_resnet50_init_reuse, self).__init__()
        self.business_layer = []

        if eval:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=nn.BatchNorm2d)
        else:
            self.backbone = get_resnet50(num_classes=19, dilation=[1, 1, 1, 2], bn_momentum=bn_momentum,
                                         is_fpn=False,
                                         BatchNorm2d=norm_layer)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2



        self.proj = Projection(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.proj.business_layer

        self.early_fusion = STAGE2(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                                   bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.early_fusion.business_layer

        self.late_fusion = STAGE3(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                                   bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.late_fusion.business_layer



    def forward(self,img, depth_mapping_3d, tsdf, sketch_gt,seg_2d,gt):

        h, w = img.size(2), img.size(3)

        feature2d = self.backbone(img) #C=12
        feature3d = self.proj(feature2d, depth_mapping_3d) #B,C,60,36,60

        early_results = self.early_fusion(feature3d, tsdf)
        late_results = self.late_fusion(feature3d, tsdf, early_results['pred_semantic'],depth_mapping_3d,gt,early_results)
        results = {'pred_semantic':early_results['pred_semantic'],
                   'pred_semantic_refine':late_results['pred_semantic_refine']}
        return results

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
