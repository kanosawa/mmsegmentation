from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot


def main():

    config = 'configs/pspnet/pspnet_r50-d8_480x360_live2d.py'
    checkpoint = 'work_dirs/live2d/latest.pth'
    img = 'data/live2d/train/hiyori.png'
    opacity = 0.5
    out_file = 'result.jpg'

    palette = [
        [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]
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
