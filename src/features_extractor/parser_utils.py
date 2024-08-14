from __future__ import annotations
import re
import logging
import ast


def no_exception_eval(s):
    try:
        return ast.literal_eval(s)
    except:
        return s


def get_func_from_str(func_repre:str)->tuple[str, list, dict]:
    func_repre = func_repre.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    m = re.match("^([^\(|^\)]*)(\((.*)\))*$", func_repre)
    if m:
        func_name = m.group(1)
        params = m.group(3)
        return func_name, *parse_params(params)
    logging.warning(f"String {func_repre} is not a valid func repre")
    return None, None, None


def parse_params(params_str:str)->tuple[list, dict]:
    args = []
    kwargs = dict()
    if params_str is not None:
        for param in split_with_ignore_regex(params_str, mask_regex='\[(.*)\]'):
            if "=" in param:
                name, value = param.split("=")
                kwargs[name] = no_exception_eval(value)
            else:
                args.append(no_exception_eval(param))
    return args, kwargs





def split_with_ignore_regex(string2split:str, mask_regex:str='\((.*)\)', split_by:str=","):
    """
    Splits a string by a specified delimiter, while ignoring substrings that match a given regex pattern. 

    The function will exclude any substring that matches the provided regex from being split. 
    Substrings that match the regex can contain the delimiter, but they will be preserved in the final split output.

    Args: string2split (str): The string to be split. The function assumes that this string does not contain the
    following characters: #, $. (TODO: Add support for specifying characters to be excluded from splitting).
    mask_regex (str): A regex pattern. Substrings matching this pattern will be ignored during the split operation.
    split_by (str): The delimiter character used to split the string. The value of this argument cant be # or $.

    Returns:
        List[str]: A list of substrings resulting from the split operation.
    """
    masks = dict()
    masked_repre = string2split
    for i, m in enumerate(re.finditer(mask_regex, string2split)):
        masks[f"#{i}$"] = m.group(0)
        s,e = m.span()
        masked_repre = masked_repre[:s] + f"#{i}$" + masked_repre[e:]
    split_repre = masked_repre.split(split_by)
    result = []
    for repre in split_repre:
        new_repre = repre
        for m in re.finditer("#[\d]+\$", repre):
            s,e = m.span()
            new_repre = new_repre[:s] + masks[m.group(0)] + new_repre[e:]
        result.append(new_repre.strip())

    return result


        

                    

