"""Prepare the base DP/dynamics pair by composing the two training atoms."""
import shutil

from . import dp_train, dyn_train
from .common import Context, finish, parser


def prepare(ctx):
    def adopt(key, source):
        source = ctx.path(source)
        ctx.require(source)

        def action(work):
            dest = work / source.name
            if not ctx.dry:
                shutil.copy2(source, dest)
            else:
                print(f"DRY_RUN: copy {source} -> {dest}")
            return {key: str(dest), "artifacts": [str(dest)]}
        return ctx.stage(ctx.root / "base" / key, {"source": str(source)}, action)

    dp = (adopt("dp", ctx.c["base_dp"]) if ctx.c.get("base_dp") else
          dp_train.train(ctx, ctx.root / "base/dp", base=True))
    dyn = (adopt("dyn", ctx.c["base_dyn"]) if ctx.c.get("base_dyn") else
           dyn_train.train(ctx, ctx.root / "base/dyn", dp["dp"], base=True))
    return {"dp": dp["dp"], "dyn": dyn["dyn"]}


def main():
    p = parser(__doc__)
    args = p.parse_args()
    finish(prepare(Context.from_args(args)))


if __name__ == "__main__":
    main()
