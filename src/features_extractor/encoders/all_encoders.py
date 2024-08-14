import logging
from abc import ABC, abstractmethod
import re
import pandas as pd
import math
from features_extractor.encoders.encoder_factory import register_encoder
import ast
from typing import Union
import numpy as np
import features_extractor.parser_utils






@register_encoder
class Encoder(ABC):
    """
    A simple encoder that defines the encoder's API. Its main functionalities are:

    - Calculate encoding parameters from the given data using the `fit` function.
    - Encode a given series into multiple columns using the `__call__` function.
    """



    def fit(self, to_encode:pd.Series)->None:
        """
        Generates new encoding variables by fitting the encoder to the training data. These new variables can then be
        used in the encoding process when encoding data.
        Args:
            to_encode: The training data to be fitted to.
        """
        pass

    def __call__(self, to_encode:pd.Series)->pd.DataFrame:
        """
        Encode the series into multiple columns using the encoding variables.
        This function should not modify the original series but instead create new columns with the encoded data.

        Args:
            to_encode (pd.Series): The data to encode. Each encoder may expect a different type of dtype.

        Returns:
            pd.DataFrame: The encoded data.
        """
        return pd.DataFrame(data=to_encode)


    def __repr__(self) -> str:
        """

        Returns: A string representation of the encoder. This representation should adhere to the parsing structure
        defined in `parser_utils.py` and include all the arguments necessary to recreate the encoder with all its
        attributes.

        """
        return f"{self.__class__.__name__}"




class CyclicEncoder(Encoder):
    """
    This is a class for encoders with cyclic behavior. Encoders that extend this class will implement the
    `encode_time` function, which expects a series as input and outputs an encoded series. The encoded series
    is then converted into sine and cosine columns according to the cycle duration.
    """

    def __init__(self, cycle:int):
        """
        Args:
            cycle: The cycle duration.
        """
        super(CyclicEncoder, self).__init__()
        self.cycle = cycle

    @abstractmethod
    def encode_time(self, to_encode:pd.Series)->pd.Series:
        """
        Encodes the time into one series without taking into account the cycling behaviour.
        Args:
            to_encode: data to be encoded.

        Returns: the encoded data

        """
        pass

    def __call__(self, to_encode:pd.Series)->pd.DataFrame:
        """

        Args:
            to_encode: Series to encode

        DataFrame: A DataFrame with two columns:
                - '<encoder_name>|cos': Encodes the cyclic behavior of the time encoder using the cosine function.
                - '<encoder_name>|sin': Encodes the cyclic behavior of the time encoder using the sine function.
        """
        encoded_col = self.encode_time(to_encode)
        sin_col = np.sin(2 * np.pi * encoded_col/self.cycle)
        cos_col = np.cos(2 * np.pi * encoded_col/self.cycle)
        result = pd.DataFrame(data={f"{encoded_col.name}|sin":sin_col, f"{encoded_col.name}|cos":cos_col}, index=to_encode.index)
        return result




        

@register_encoder
class InRange(Encoder):
    """
    Replace all values that smaller than 'min_val' or larger than 'max_val' with None.
    """
    def __init__(self, min_val=0, max_val=math.inf):
        super(InRange, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, to_encode:pd.Series):
        if self.min_val is None:
            self.min_val = to_encode.min()
        if self.max_val is None:
            self.max_val = to_encode.max()
        

    def __call__(self, to_encode: pd.Series):
        encoded_col = to_encode.copy()
        encoded_col.loc[to_encode < self.min_val] = pd.NA
        encoded_col.loc[to_encode > self.max_val] = pd.NA
        return pd.DataFrame(data={self.__repr__(): encoded_col}, index=to_encode.index)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_val={self.min_val},max_val={self.max_val})"


@register_encoder
class Positive(InRange):
    """
    Replace all the negative values to None.
    """
    def __init__(self):
        super(Positive, self).__init__(min_val=0, max_val=math.inf)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    
@register_encoder
class LengthOfList(Encoder):
    """
    Encode a Series of lists to a Series of their respective lengths.
    """
    def __call__(self, to_encode: pd.Series):
        try:
            encoded_column = to_encode.apply(lambda x: len(x.split(",")))
        except:
            encoded_column = to_encode.apply(lambda x: len(x))
        return pd.DataFrame(data={self.__repr__(): encoded_column}, index=to_encode.index)
        
@register_encoder
class LengthOfString(Encoder):
    """
    Encode a Series of strings to a Series of their respective lengths.
    """
    def __call__(self, to_encode: pd.Series):
        encoded_column = to_encode.apply(lambda x: len(x))
        return pd.DataFrame(data={self.__repr__(): encoded_column}, index=to_encode.index)
    

@register_encoder
class IsEmpty(Encoder):
    """
    Encode a Series of strings to boolean values. True if the string consists of only whitespaces and false otherwise
    """
    def __init__(self, empty_value_regex="^[\s]*$"):
        super(IsEmpty, self).__init__()
        self.empty_value_regex = empty_value_regex

    def __call__(self, to_encode: pd.Series):

        encoded_column =  to_encode.apply(lambda x: re.match(self.empty_value_regex, x) is not None)
        
        return pd.DataFrame(data={self.__repr__(): encoded_column}, index=to_encode.index)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(empty_value_regex={self.empty_value_regex})"

@register_encoder
class DayOfWeek(CyclicEncoder):
    """
    Encode a Series of times into their corresponding day of the week values. This encoder accounts for the cyclical
    nature of the week by generating both sine and cosine representations of the day of the week.
    """
    def __init__(self):
        super(DayOfWeek, self).__init__(cycle=7)

    def encode_time(self, to_encode: pd.Series):
        encoded_column =  pd.to_datetime(to_encode, format='ISO8601').dt.day_of_week
        encoded_column.name=self.__repr__()
        return encoded_column

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

@register_encoder
class Hour(CyclicEncoder):
    """
    Encode a Series of times into their corresponding hour values. This encoder accounts for the cyclical
    nature of the day by generating both sine and cosine representations of the hour.
    """
    def __init__(self):
        super(Hour, self).__init__(cycle=24)

    def encode_time(self, to_encode: pd.Series):
        encoded_column =  pd.to_datetime(to_encode, format='ISO8601').dt.hour
        encoded_column.name=self.__repr__()
        return encoded_column

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

@register_encoder
class CategoryOneHot(Encoder):
    """
    Encode a series of categorical values into boolean columns, where each column represents a different category.
    A column will have a positive value (True) only if the corresponding row contains that category.
    """
    def __init__(self, categories:list=None, invalid_category_repre="other", valid_category_regex=None, multi_categories=False):
        """
        Args:
            categories: A list of categories to consider for encoding. If None, the encoder will collect all unique values
                from the given series when calling the fit function and use these values as the encoder's categories.
            invalid_category_repre: The representation to assign to rows that do not match any of the specified categories.
            valid_category_regex: A regular expression to filter categories. If not None, only categories matching this regex
                will be considered when collecting categories in the fit function.
                multi_categories: True is the dtype of the data is a list of categories
        """
        super(CategoryOneHot, self).__init__()
        self.categories = categories 
        self.invalid_category_repre = invalid_category_repre
        self.valid_category_regex = valid_category_regex
        self.multi_categories = multi_categories

    def fit(self, to_encode:pd.Series):
        if self.categories is None:
            if self.multi_categories:
                to_encode = to_encode.dropna()
                not_empty = to_encode.loc[to_encode.apply(lambda x: len(x) > 0)]
                self.categories = np.unique(np.concatenate(not_empty.values)).tolist() if len(not_empty) > 0 else []
            else:

                self.categories = pd.unique(to_encode).tolist()
        filtered_categories = set()
        if self.valid_category_regex is not None:
            for cat in self.categories:
                if type(cat)==str:
                    m = re.match(self.valid_category_regex, cat)
                    if m:
                        filtered_categories.append(cat) 
            self.categories = list(filtered_categories)

    def fill_invalid(self, categories:list):
        valid_categories = []
        for cat in categories:
            if cat in self.categories:
                valid_categories.append(cat)
        if len(valid_categories) == 0:
            valid_categories.append(self.invalid_category_repre)
        return valid_categories

    def __call__(self, to_encode: pd.Series):
        to_encode = to_encode.apply(lambda x: x if isinstance(x, list) else [x])
        to_encode = to_encode.apply(lambda x: self.fill_invalid(x))
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=self.categories+[self.invalid_category_repre])
        encoded_col = mlb.fit_transform(to_encode)
        result = pd.DataFrame(encoded_col, columns=[self.__repr__()+"|"+str(cat) for cat in mlb.classes_], index=to_encode.index)

        return result
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"categories={self.categories},"
                f"invalid_category_repre={self.invalid_category_repre},"
                f"valid_category_regex={self.valid_category_regex}, "
                f"multi_categories={self.multi_categories})")


@register_encoder
class ContainRegex(Encoder):
    """
    Encode a Series of strings to boolean values. True if the string match the given regex or not.
    """
    def __init__(self, regex):
        super(ContainRegex, self).__init__()
        self.regex = regex

    def __call__(self, to_encode: pd.Series):
        encoded_col = to_encode.apply(lambda x: re.match(self.regex, x) is not None)
        return pd.DataFrame(data={self.__repr__(): encoded_col}, index=to_encode.index)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(regex={self.regex})"
    
@register_encoder
class HasLabels(Encoder):
    """
    Encode a Series of strings to boolean values. True if the string match any label's regex.
    """
    def __init__(self, additional_regex=[]) -> None:
        """
        Args:
            additional_regex: Additional regex to consider
        """
        super(HasLabels, self).__init__()
        self.additional_regex = set(additional_regex)
        regex_list = {"\[.*\]", "^([A-Z|a-z]+:)+.*$"}
        regex_list.update(self.additional_regex)
        self.encoders = [ContainRegex(l) for l in regex_list]

    def __call__(self, to_encode: pd.Series):
        all_encoded_values = []
        for encoder in self.encoders:
            all_encoded_values.append(encoder(to_encode).values[:, 0])
        encoded_col = np.logical_or.reduce(all_encoded_values)
        return pd.DataFrame(data={self.__repr__(): encoded_col}, index=to_encode.index)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(additional_regex={list(self.additional_regex)})"
    

@register_encoder
class HasEmoji(Encoder):
    """
    Encode a Series of strings to boolean values. True if the string contains emoji.
    """
    from emoji import UNICODE_EMOJI
    english_emojies = UNICODE_EMOJI['en']

    def __init__(self):
        super(HasEmoji, self).__init__()

    def _contain_emoji(self, s):
        for c in s:
            if c in HasEmoji.english_emojies:
                return True
        return False

    def __call__(self, to_encode: pd.Series):
        return pd.DataFrame(data={self.__repr__(): to_encode.apply(lambda x: self._contain_emoji(x))}, index=to_encode.index)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

