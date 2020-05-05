import torch
import torch.nn as nn
import Pix2PixGAN.Generator as pix2pixG
import Pix2PixGAN.Discriminator as pix2pixD
import Pix2PixGAN.Initialization as pix2pixInit
import Pix2PixGAN.GANLoss as pix2pixLoss
import functools
import Dataset.RGBDFaceDataset as rgbdDataset
import Utils.Visualization as Vis
from tqdm import tqdm
from os import path

if __name__ == '__main__':

    ### Define Networks ###

    netG = pix2pixG.UnetGenerator(input_nc=4, output_nc=4, num_downs=8, ngf=64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout=True)
    netG = pix2pixInit.init_net(netG, gpu_ids=[0])

    netD = pix2pixD.NLayerDiscriminator(input_nc=8, ndf=64, n_layers=3, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True))
    netD = pix2pixInit.init_net(netD, gpu_ids=[0])

    ### Load Exsting Model State ###

    if path.exists("Result/trainedGenerator.pth") and path.exists("Result/trainedDiscriminator.pth"):

        netG.load_state_dict(torch.load("Result/trainedGenerator.pth"))
        netD.load_state_dict(torch.load("Result/trainedDiscriminator.pth"))

        startEpoch = int(input('Enter startEpoch:'))
        print('startEpoch:', startEpoch, type(startEpoch))

    else:
        startEpoch = 1

    ### Training settings ###

    learningRate = 0.0002
    lambda_L1 = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterionGAN = pix2pixLoss.GANLoss("vanilla").to(device)
    criterionL1 = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=learningRate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=learningRate, betas=(0.5, 0.999))

    lambda_rule = lambda epoch: 1.0 - max(0, epoch + startEpoch - 100) / float(100 + 1)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    ### Training ###

    dataset = rgbdDataset.RGBDFaceDataset(imageSize=256, path="Dataset/")
    dataset = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for epoch in range(startEpoch, 200 + 1):
        print("Epoche: ", epoch, "(LearningRates:", scheduler_G.get_last_lr(), scheduler_D.get_last_lr(),")")
        for i, data in enumerate(tqdm(dataset)):

            heatmap = data['Heatmap'].to(device)
            realRGBD = data['RGBD'].to(device)
            #Vis.showDatapair(realRGBD[0], heatmap[0])
            fakeRGBD = netG(heatmap)
            #Vis.showDatapair(fakeRGBD[0], heatmap[0])


            ### Update Discriminator ###  similar to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py

            optimizer_D.zero_grad()  # set D's gradients to zero
            """Calculate GAN loss for the discriminator"""
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((heatmap, fakeRGBD),1)
            pred_fake = netD(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((heatmap, realRGBD), 1)
            pred_real = netD(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()  # update D's weights

            ### Update Generator ###  similar to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py

            optimizer_G.zero_grad()  # set G's gradients to zero
            """Calculate GAN and L1 loss for the generator"""
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((heatmap, fakeRGBD), 1)
            with torch.no_grad():
                pred_fake = netD(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True)
            # Second, G(A) = B
            loss_G_L1 = criterionL1(fakeRGBD, realRGBD) * lambda_L1
            # combine loss and calculate gradients
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()  # udpate G's weights

        ### Learning Rate Schedular

        scheduler_G.step()
        scheduler_D.step()

        ### Save Modell ###

        torch.save(netG.cpu().state_dict(), "Result/trainedGenerator.pth")
        torch.save(netD.cpu().state_dict(), "Result/trainedDiscriminator.pth")

        ### Trace Modell ###

        noise = torch.randn(heatmap.shape)
        traced = torch.jit.trace(netG.eval(), noise)
        netG.train().to(device)
        netD.train().to(device)
        traced.save('Result/tracedGenerator.zip')
        #print("LoadModel")
        #loaded = torch.jit.load('trainedGenerator.zip')
        #print(loaded)
        #print(loaded.code)
        #temp = loaded.forward(heatmap)
        #Vis.showDatapair(temp[0],heatmap[0])

        ### Export Sample Image ###

        Vis.exportExample(fakeRGBD[0], heatmap[0], "Result/example.png")
