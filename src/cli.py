import argparse
from pathlib import Path
from dataclasses import fields, MISSING
from main import translate_video, TranslateSettings

from typing import ClassVar, Protocol, Any, Type

class IsDataclass(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[dict[str, Any]]


def set_args_from_dataclass(parser: argparse.ArgumentParser, klass: Type[IsDataclass]):
    for arg in fields(klass):
        kwargs = {}

        if arg.default is MISSING:
            kwargs['required'] = True
        else:
            kwargs['default'] = arg.default

        try:
            kwargs['help'] = arg.type.__metadata__[0]
        except Exception:
            pass

        arg_name = arg.name.lower().replace("_", "-")

        if arg.default is True:
            kwargs['action'] = 'store_false'
        elif arg.default is False:
            kwargs['action'] = 'store_true'
        else:
            kwargs['type'] = arg.type

        parser.add_argument(
            f'--{arg_name}',
            **kwargs
        )

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        help='Path to the video to translate'
    )
    set_args_from_dataclass(parser, TranslateSettings)
    return parser

def get_dataclass_from_args(args: argparse.Namespace, klass: Type[IsDataclass]):
    args_dict = vars(args)
    kwargs = {}
    for k, v in args_dict.items():
        name = k.upper()
        if name not in (f.name for f in fields(klass)):
            continue
        kwargs[k.upper()] = v
    return klass(**kwargs)

def translate() -> None:
    parser = get_parser()
    args = parser.parse_args()
    settings = get_dataclass_from_args(args, TranslateSettings)
    print(f"translating using settings {settings}")
    translate_video(Path(args.input), settings)

if __name__ == '__main__':
    translate()
