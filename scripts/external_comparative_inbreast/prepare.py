"""Create the image-only INbreast input and checkpoint protocol."""
from common import prepare


if __name__ == '__main__':
    context = prepare()
    print(f"prepared images={len(context['rows'])} checkpoints={len(context['checkpoints'])} signature={context['signature']}")
