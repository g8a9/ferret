def parse_explainer_args(explainer_args):
    init_args = explainer_args.get("init_args", {})
    call_args = explainer_args.get("call_args", {})
    return init_args, call_args