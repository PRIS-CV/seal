import torch
from tqdm import tqdm

from info.utils.utils import load_to_device


def extract_visual_prototype(model, dataloader, device, num, dim, f_prototype):
    
    visual_prototypes = torch.zeros((num, dim), dtype=torch.float32).to(device)
    f_num = torch.zeros((num, ), dtype=torch.long).to(device)

    pbar = tqdm(dataloader)
    pbar.set_description(f'Extract Visual Features')
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(pbar):
            data = load_to_device(data, device)
            features = model.get_visual_feature(data)    # q2l [64, 620, 2048]
            targets = data['t']
             
            for i, (f, t) in enumerate(zip(features, targets)):
                visual_prototypes[t == 1] += f[t==1]
                f_num[t == 1] += 1

    visual_prototypes /= f_num.unsqueeze(1)
    visual_prototypes = visual_prototypes.cpu()
    print("Visual Prototypes Build Success! Save to {}".format(f_prototype))
    torch.save(visual_prototypes, f_prototype)
