import inspect
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Optional

import pytorch_lightning as pl


@dataclass(frozen=True)
class LitArg:
    """Dataclass to represent init args of an object
    """

    name: str
    types: tuple
    default: Any
    required: bool = False
    context: Optional[str] = None


class LightningArgumentParser(ArgumentParser):
    """
    Extension of argparse.ArgumentParser that lets you parse arbitrary object init args.

    Example::

        from pl_bolts.utils.arguments import LightningArgumentParser

        parser.add_datamodule_args(MyDataModule)
        parser.add_model_args(MyModel)
        parser.add_trainer_args()
        args = parser.parse_lit_args()
    """

    def __init__(self, *args, ignore_required_init_args=True, **kwargs):
        """
        Args:
            ignore_required_init_args (bool, optional): Whether to include positional args when adding
            object args. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        self.ignore_required_init_args = ignore_required_init_args

        self._default_obj_args = dict()
        self._added_arg_names = []

    def add_object_args(self, name: str, obj: object, root_cls: object = None):
        """Add any arbitrary class's init args to parser under grouped Namespace.

        Args:
            name (str): Name of the object group you want to add.
            obj (object): Any arbitrary object you want arguments for. Must have type annotations in __init__ signature.
            root_cls (object, optional): Parse args up inheritence tree until you hit this object. Defaults to None.
        """
        default_args = gather_lit_args(obj, root_cls)
        self._default_obj_args[name] = default_args
        for arg in default_args:
            if arg.name in self._added_arg_names:
                continue
            self._added_arg_names.append(arg.name)
            kwargs = dict(type=arg.types[0])
            if arg.required and not self.ignore_required_init_args:
                kwargs["required"] = True
            else:
                kwargs["default"] = arg.default
            self.add_argument(f"--{arg.name}", **kwargs)

    def add_model_args(self, model_obj: pl.LightningModule):
        """Add arguments from model_obj to the parser

        Args:
            model_obj (pl.LightningModule): Any LightningModule subclass
        """
        self.add_object_args('model', model_obj, pl.LightningModule)

    def add_datamodule_args(self, datamodule_obj):
        """Add arguments from datamodule_obj to the parser

        Args:
            datamodule_obj (pl.LightningDataModule): Any LightningDataModule subclass
        """
        self.add_object_args('datamodule', datamodule_obj, pl.LightningDataModule)

    def add_trainer_args(self, trainer_obj: pl.Trainer = pl.Trainer):
        """Add Lightning's Trainer args to the parser.

        Args:
            trainer_obj (pl.Trainer, optional): The trainer object to add arguments from. Defaults to pl.Trainer.
        """
        self.add_object_args('trainer', trainer_obj, pl.Trainer)

    def parse_lit_args(self, *args, **kwargs):
        parsed_args_dict = vars(self.parse_args(*args, **kwargs))
        lit_args = Namespace()
        for name, default_args in self._default_obj_args.items():
            lit_obj_args = dict()
            for arg in default_args:
                arg_is_member_of_obj = arg.name in parsed_args_dict
                arg_should_be_added = not arg.required or (arg.required and not self.ignore_required_init_args)
                if arg_is_member_of_obj and arg_should_be_added:
                    lit_obj_args[arg.name] = parsed_args_dict[arg.name]
            lit_args.__dict__.update(**{name: Namespace(**lit_obj_args)})
        return lit_args


def gather_lit_args(cls, root_cls=None):

    if root_cls is None:
        if issubclass(cls, pl.LightningModule):
            root_cls = pl.LightningModule
        elif issubclass(cls, pl.LightningDataModule):
            root_cls = pl.LightningDataModule
        else:
            root_cls = cls

    blacklisted_args = ["self", "args", "kwargs"]
    arguments = []
    argument_names = []
    for obj in inspect.getmro(cls):

        if obj is root_cls and len(arguments) > 0:
            break

        if issubclass(obj, root_cls):

            default_params = inspect.signature(obj.__init__).parameters

            for arg in default_params:
                arg_type = default_params[arg].annotation
                arg_default = default_params[arg].default

                try:
                    arg_types = tuple(arg_type.__args__)
                except AttributeError:
                    arg_types = (arg_type,)

                # If type is empty, that means it hasn't been given type hint. We skip these.
                arg_is_missing_type_hint = arg_types == (inspect._empty,)
                # Some args should be ignored by default (self, kwargs, args)
                arg_is_in_blacklist = arg in blacklisted_args and arg_is_missing_type_hint
                # We only keep the first arg we see of a given name, as it overrides the parents
                arg_is_duplicate = arg in argument_names
                # We skip any of the above 3 cases
                do_skip_this_arg = arg_is_in_blacklist or arg_is_missing_type_hint or arg_is_duplicate

                # Positional args have no default, but do have a known type or types.
                arg_is_positional = arg_default == inspect._empty and not arg_is_missing_type_hint
                # Kwargs have both a default + known type or types
                arg_is_kwarg = arg_default != inspect._empty and not arg_is_missing_type_hint

                if do_skip_this_arg:
                    continue

                elif arg_is_positional or arg_is_kwarg:
                    lit_arg = LitArg(
                        name=arg,
                        types=arg_types,
                        default=arg_default if arg_default != inspect._empty else None,
                        required=arg_is_positional,
                        context=obj.__name__,
                    )
                    arguments.append(lit_arg)
                    argument_names.append(arg)
                else:
                    raise RuntimeError(
                        f"Could not determine proper grouping of argument '{arg}' while gathering LitArgs"
                    )
    return arguments
