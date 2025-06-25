class Sameness_Loss(nn.Module):
    def __init__(self):
        super(Sameness_Loss12, self).__init__()


    def forward(self, output, label_weight, label):
        bs = output.shape[0]
        loss = torch.tensor(0.).cuda()
        label_weight_mask = label_weight.int().reshape(bs,-1)
        output = output.permute(0, 2, 3, 4, 1).contiguous().view(bs,-1, 12)

        for b in range(bs):
            label_b = label[b]
            weight_mask = label_weight_mask[b]
            output_b = output[b]
            class_loss = torch.tensor(0.).cuda()
            region_num=0
            for i in range(1,12):
                region_mask = label_b == i
                mask = region_mask.int() & weight_mask
                if mask.sum() == 0:
                    continue
                mask = torch.nonzero(mask).view(-1)
                filterOutput = torch.index_select(output_b, 0, mask)
                filterOutput = F.softmax(filterOutput, dim=1)
                probs = torch.mean(filterOutput, dim=0)
                most = torch.argmax(probs)
                if most.item() == i:
                    local_loss = Categorical(probs=probs).entropy()
                    class_loss = class_loss + local_loss
                    region_num += 1
            if region_num == 0:
                continue
            loss = loss + class_loss/region_num
        loss = loss/bs
        return loss
