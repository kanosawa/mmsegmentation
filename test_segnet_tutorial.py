from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot


def main():

    config = 'configs/pspnet/pspnet_r50-d8_480x360_segnet_tutorial.py'
    checkpoint = 'work_dirs/latest.pth'
    img = 'data/segnet_tutorial/train/0001TP_006690.png'
    opacity = 0.5
    out_file = 'result.jpg'

    palette = [
        [128, 128, 128], [128, 0, 0], [192, 192, 128], [255, 69, 0], [128, 64, 128], [60, 40, 222],
        [128, 128, 0], [192, 128, 128], [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192]
    ]

    model = init_segmentor(config, checkpoint, device='cuda:0')
    result = inference_segmentor(model, img)

    show_result_pyplot(
        model,
        img,
        result,
        palette,
        opacity=opacity,
        out_file=out_file
    )


if __name__ == '__main__':
    main()
