import logging
from DetectionUtilities.feature_extractor import parser_utils
import pandas as pd
import re

ENCODED_COL_STRUCTURE_REGEX = "^([^|]*)\|([^|]*)\|.*"


encoders_registery = dict()



def register_encoder(cls):
    encoders_registery[cls.__name__.lower()] = cls
    return cls

def get_encoder(name):
    """
    Args:
        name (str): The class name of the required encoder.

    Returns:
        type: The encoder class corresponding to the given class name.
    """
    if name.lower() in encoders_registery:
        return encoders_registery[name.lower()]
    
    

def get_encoder_from_str(encoder_str):
    """
    Create an encoder object from its string representation.

    Args:
        encoder_str (str): A string representation of one or more encoders, where multiple representations are separated by commas. The string should not contain the characters '#' or '$'. The encoder structure is `<encoder_class_name>(<constructor_args_and_kwargs_separated_by_commas>)`.

    Returns:
        list: A list of encoder objects created from the given string representations.
    """
    func_repres = parser_utils.split_with_ignore_regex(encoder_str)
    encoders = []

    for repre in func_repres:
        encoder_name, args, kwargs = parser_utils.get_func_from_str(repre)
        encoder = get_encoder(encoder_name)
        if encoder is not None:
            encoders.append(encoder(*args, **kwargs))
        else:
            logging.warning(f"[EncoderFactory] encoder cant be generated for {repre}")
    return encoders

def build_encoders_dict_from_metadata(metadata_df:pd.DataFrame, names_column:str, encoder_column:str):
    """
    This function creates a list of encoders for each column based on the metadata.

    Args:
        metadata_df (pd.DataFrame): A DataFrame containing the columns specified in `names_column` and `encoder_column`.
        names_column (str): The name of the column in `metadata_df` that contains the relevant names for which encoders will be generated.
        encoder_column (str): The name of the column in `metadata_df` that contains a list of encoder representations for each name in the `names_column` column.

    Returns:
        dict: A dictionary where each key is a name from `names_column` and the corresponding value is a list of encoders.
    """
    column_name, encoder_objs = zip(*metadata_df[[names_column, encoder_column]].apply(lambda x: (x[names_column], get_encoder_from_str(x[encoder_column]) if pd.notna(x[encoder_column]) else None), axis=1))
    return dict(zip(column_name, encoder_objs))



def run_encoders_from_metadata(to_encode:pd.DataFrame, metadata_df:pd.DataFrame, names_column:str, encoder_column:str, fit_and_normalize=False, normalize=False)->pd.DataFrame:
    """
    Encodes a DataFrame according to the provided metadata file. This function creates a list of encoders for each column
    based on the metadata, then applies the encoder to generate encoded columns. If a column is not found in the
    metadata or does not have a corresponding encoders, the function will log a warning and include the original
    column in the result.

    Args:
        to_encode (pd.DataFrame): The data to encode.
        metadata_df (pd.DataFrame): A DataFrame containing the columns specified in `names_column` and `encoder_column`.
        names_column (str): The name of the column in `metadata_df` that contains the names of the columns in `to_encode`
            that should be encoded.
        encoder_column (str): The name of the column in `metadata_df` that contains the corresponding encoder representations
            for each column to encode.
        fit_and_normalize (bool): If `True`, after encoding a column, the encoder will run `encoder.fit_and_normalize`
            on the encoded data.
        normalize (bool): If `True`, after encoding a column, the encoder will run `encoder.normalize` on the encoded data.

    Returns:
        pd.DataFrame: The encoded data.
    """
    col2encoders = build_encoders_dict_from_metadata(metadata_df, names_column, encoder_column)
    result = []
    for feature_col in to_encode:
        if feature_col in col2encoders:
            encoders = col2encoders[feature_col]
            if encoders is not None:
                for encoder in encoders:
                    print(encoders, feature_col)

                    encoder.fit(to_encode[feature_col])

                    encoded_df = encoder(to_encode[feature_col])
                    remaned_cols = {encoder_name: feature_col + "|" + encoder_name for encoder_name in encoded_df.columns}
                    encoded_df.rename(columns=remaned_cols, inplace=True)
                    if fit_and_normalize:
                        encoded_df = encoder.fit_and_normalize(encoded_df)
                    elif normalize:
                        encoded_df = encoder.normalize(encoded_df)
                    result.append(encoded_df)
            else:
                logging.warning(f"[Encoder] col {feature_col} doesnt have encoder in metadata - add the original column")
                result.append(to_encode[[feature_col]])
        else:
            logging.warning(f"[Encoder] columns {feature_col} doesnt exist in encoding metadata - add the original column")
            result.append(to_encode[[feature_col]])
    return pd.concat(result, axis=1)