from collections import defaultdict

import torch
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors


class DetectionPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter.
    def __init__(
        self,
        *,
        data_augmentation,
        hflip_prob=0.5,
        mean=(123.0, 117.0, 104.0),
        backend="pil",
        _=True,
    ):

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        if data_augmentation == "hflip":
            transforms += [T.RandomHorizontalFlip(p=hflip_prob)]
        elif data_augmentation == "multiscale":
            transforms += [
                T.RandomShortestSize(min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssd":
            fill = defaultdict(lambda: mean, {tv_tensors.Mask: 0})
            transforms += [
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=fill),
                #T.Lambda(lambda y: print("DEBUG:", type(y))),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "ssdlite":
            transforms += [
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
            ]
        elif data_augmentation == "custom":
            fill = defaultdict(lambda: mean, {tv_tensors.Mask: 0})

            transforms += [
                T.Resize((400,600)),
                T.RandomGrayscale(p=1),
                T.RandomRotation(180, expand=True),
                T.Resize((400,600)),
                T.RandomChoice([
                    T.RandomPhotometricDistort(),
                    T.JPEG((50,100)),
                ]), 
                T.RandomApply([
                    T.Resize((400,600)),
                    T.RandomChoice([
                        T.RandomPerspective(distortion_scale=0.1),
                    ]),                  
                ], p=0.67),    
                T.RandomApply([
                    T.Pad(20),
                    T.ElasticTransform(), 
                    T.Resize((400,600)),
                ], p=0.2),   
                T.RandomApply([
                    T.RandomZoomOut(fill=fill),
                    T.RandomIoUCrop(),
                ], p=0.8),   
                T.RandomHorizontalFlip(p=hflip_prob),
                T.Resize((400,600)),
            ]
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

        transforms += [T.PILToTensor()]

        transforms += [T.ToDtype(torch.float, scale=True)]

        transforms += [
            T.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
            T.SanitizeBoundingBoxes(),
            T.ToPureTensor(),
        ]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self, backend="pil", _=True):
        transforms = []
        backend = backend.lower()
        
        if backend in ["tensor", "pil"]:
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        else:
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        transforms += [T.ToDtype(torch.float, scale=True)]

        transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
