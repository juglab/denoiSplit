import torch
import torch.nn.functional as F


def compute_exclusion_loss(img1, img2, level=3):
    loss_gradx, loss_grady = compute_exclusion_loss_vector(img1, img2, level=3)
    loss_gradxy = torch.sum(loss_gradx) / 3. + torch.sum(loss_grady) / 3.
    return loss_gradxy / 2


def compute_exclusion_loss_vector(img1, img2, level=3):
    gradx_loss = []
    grady_loss = []

    for l in range(level):
        gradx1, grady1 = compute_gradient(img1)
        gradx2, grady2 = compute_gradient(img2)

        alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
        alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))

        gradx1_s = (torch.sigmoid(gradx1) * 2) - 1
        grady1_s = (torch.sigmoid(grady1) * 2) - 1
        gradx2_s = (torch.sigmoid(gradx2 * alphax) * 2) - 1
        grady2_s = (torch.sigmoid(grady2 * alphay) * 2) - 1

        prod = torch.multiply(torch.square(gradx1_s), torch.square(gradx2_s))
        prod = prod.view((len(prod), -1))
        gradx_loss.append(torch.mean(prod, dim=1)**0.25)

        prod = torch.multiply(torch.square(grady1_s), torch.square(grady2_s))
        prod = prod.view((len(prod), -1))
        grady_loss.append(torch.mean(prod, dim=1)**0.25)

        img1 = F.avg_pool2d(img1, 2)
        img2 = F.avg_pool2d(img2, 2)

    return torch.cat(gradx_loss), torch.cat(grady_loss)


def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :, ]
    grady = img[..., :, 1:] - img[..., :, :-1, ]
    return gradx, grady


if __name__ == '__main__':
    img1 = torch.rand((12, 1, 64, 64))
    img2 = torch.rand((12, 1, 64, 64))
    loss = compute_exclusion_loss(img1, img2)
