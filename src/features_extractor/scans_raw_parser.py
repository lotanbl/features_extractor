import pandas as pd
import json


"""
RAW parser for PP scans

This module provides functionality to parse RAW data from PP scans. It is divided into two parts:
1. General functions for parsing JSON data.
2. Functions specific to the structure of PP scan RAW data.
"""


# --- General Functions for Parsing JSON Data ---

def extract_value_from_json(json, json_path2field:list[str]):
    """
    Extracts a value from a JSON object (represented as a dictionary) based on a given path.

    The function traverses the JSON object using the provided list of keys, which represent the full path to the desired field. 

    For example, given the following inputs:
    - `json`:
      ```json
      {
        "a1": {"b1": {"c1": "d1"}},
        "a2": {"b2": "c2"},
        "a3": {"b3": {"c3": [{"e": 1}, {"e": 2}, {"e": 3}]}}
      }
      ```
    - `json_path2field = ["a1", "b1", "c1"]`

    The function will output: `"d1"`.

    If the path includes a list of items that match the path, the function will extract values from each item in the list.

    For example, given the previous `json` and `json_path2field = ["a3", "b3", "c3", "e"]`, 
    the function will output: `[1, 2, 3]`.

    Args:
        json (dict): The JSON object to parse.
        json_path2field (List[str]): A list of strings where each string represents a key in the path to the required field.

    Returns:
        The value at the end of the specified path. This may be a single value or a list of values if the path includes lists.
    """

    if type(json) == list:
        return [extract_value_from_json(item, json_path2field.copy()) for item in json]
    if len(json_path2field) == 0:
        return json
    current_field = json_path2field.pop(0)
    if current_field in json:
        return extract_value_from_json(json[current_field], json_path2field)
    return None

def apply_extraction_to_series(raw_series:pd.Series, json_path2field:list[str]):
    """
       Applies the `extract_value_from_json` function to each value in the given Series.

       This function traverses each JSON object in the Series using the provided list of keys to extract the desired field's value. 

       Args:
           raw_series (pandas.Series): A Series where each element is a JSON object (dict) to parse.
           json_path2field (List[str]): A list of strings where each string represents a key in the path to the required field.

       Returns:
           pandas.Series: A Series containing the extracted values corresponding to the path specified. 
                          If the path includes lists within the JSON, the output will be a list of extracted values.
       """
    return raw_series.apply(lambda x: extract_value_from_json(json.loads(x), json_path2field.copy()))

def filter_series_of_lists(series:pd.Series, valid_values:list[str])->pd.Series:
    """
    Filters the lists in the given Series, keeping only the valid values specified.

    This function iterates through each list in the Series and retains only the values that are present in the `valid_values` set.

    Args:
        series (pandas.Series): A Series where each element is a list of values to filter.
        valid_values (set): A set of valid values to keep in the lists.

    Returns:
        pandas.Series: A Series of lists, where each list contains only the valid values specified.
    """
    return series.apply(lambda x: None if x is None else [value for value in x if value in valid_values])

# --- Functions Specific to the Structure of PP Scan RAW Data ---

def extract_decisions_names(raw_series:pd.Series, valid_decisions:list[str]=None):
    """
     Extracts the names of valid decisions from each JSON object in the given Series.

     This function parses each JSON object in the Series to find and extract the names of decisions it contains.
     Only those decision names that are present in the `valid_decisions` set are kept.

     Args:
        raw_series (pandas.Series): A Series where each element is a JSON object (dict) representing a PP scan's RAW data.
        valid_decisions (set, optional): A set of valid decision names to filter. If `None`, the function considers all decisions as valid.


     Returns:
         pandas.Series: A Series of lists, where each list contains the names of valid decisions extracted from the corresponding JSON object.
     """
    extracted_df = apply_extraction_to_series(raw_series, ["decisions", "decision_name"])
    if valid_decisions is not None:
        valid_decisions = filter_series_of_lists(extracted_df, valid_decisions)
    return valid_decisions

def extract_evidences_names(raw_series:pd.Series, valid_evidences:list[str]=None):
    """
     Extracts the names of valid evidences from each JSON object in the given Series.

     This function parses each JSON object in the Series to find and extract the names of evidences it contains.
     Only those evidences names that are present in the `valid_evidences` set are kept.

     Args:
        raw_series (pandas.Series): A Series where each element is a JSON object (dict) representing a PP scan's RAW data.
        valid_evidences (set, optional): A set of valid evidence names to filter. If `None`, the function considers all evidences as valid.

     Returns:
         pandas.Series: A Series of lists, where each list contains the names of valid evidences extracted from the corresponding JSON object.
     """
    extracted_df = apply_extraction_to_series(raw_series, ["evidence", "name"])
    if valid_evidences is not None:
        extracted_df = filter_series_of_lists(extracted_df, valid_evidences)
    return extracted_df

def extract_attributes_df(raw_series:pd.Series, valid_attributes:list[str]=None):
    """
    Extract all attributes and their values from each JSON object in the given Series.
    If attribute as sub attribute it'll create a different column for each sub-attribute
    Args:
        raw_series (pandas.Series): A Series where each element is a JSON object (dict) representing a PP scan's RAW data.
        valid_attributes (set, optional): A set of valid attribute names to filter. If `None`, the function considers all attributes as valid.

    Returns (pandas.DataFrame): A pd.DataFrame. The columns represent a valid attribute , and the rows represents a scan.

    """
    attributes_df = pd.json_normalize(apply_extraction_to_series(raw_series, ["sample"]))
    attributes_df.set_index(raw_series.index, inplace=True)
    if valid_attributes is not None:
        attributes_df = attributes_df[list(col for col in attributes_df.columns if col in valid_attributes)]
    return attributes_df


