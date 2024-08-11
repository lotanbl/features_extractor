

# Features Extractor

This package is designed to extract features from PP scans. Currently, it supports extracting features from attributes, evidences, and decisions.

The package provides the following tools:

1. **Encoders Module**: This module includes a diverse list of encoders with a simple API that allows you to easily add more encoders. (For usage instructions, see below.)

2. **Parser Utils**: This module enables parsing of function representation strings into names, arguments, and keyword arguments. For a use-case example, refer to the file `encoder/encoder_factory.py`. This file is an encoder factory that generates a list of encoders with their parameters from a string. For example, the string `"IsEmpty, HasLabels, HasEmoji, LengthOfString, CategoryOneHot(categories=['ERROR','FAIL','PASS'])"` will be parsed into a list of the following encoders: `IsEmpty()`, `HasLabels()`, `HasEmoji()`, `LengthOfString()`, and `CategoryOneHot(categories=['ERROR','FAIL','PASS'])`.

3. **scans_raw_parser**: This module facilitates the parsing of RAW representations in perception point scans.




**For use-case scenarios, see the file `example.py`.**


## More information about the module functionalities:

### Parser Utils

1. **Function Representation Structure**: The parser expects the given string to have the following structure:
    ```
    <any_char_except_'('_or')'>(<any_char>)
    ```

2. **Splitting Multiple Function Representations**: To split multiple function representations separated by commas, use the function `split_funcs(func_repre: str, mask_regex='\((.*)\)')`. This function masks all items in the given string that match the provided regex and then splits the string by commas.

### Encoder Module

1. **Create a New Encoder Class**: Add a new encoder class in the `all_encoders.py` file. The class should extend the abstract class `Encoder`. Ensure to add the `@register_encoder` decorator above the class definition to register the encoder in the encoder factory.

2. **CyclicEncoder API**: The package also provides the `CyclicEncoder` API for encodings with cyclic behavior. For example, `DayOfWeek` extends this API. It encodes each sample's day, and the `CyclicEncoder` converts this encoding into cosine and sine features based on a 7-day cycle.

3. **Building an Encoder from a String Representation**:
    - **Build an encoder from a string representation**:
        1. Use the function `get_encoder_from_str` from the `encode_factory.py` file.
        2. Each encoder should implement the `__repr__` function in a way that allows the encoder factory to generate encoders from it.
        3. A valid encoder representation should not contain the characters '#' or '$', and must adhere to the following structure: `<encoder_class_name>(<constructor_args_and_kwargs_separated_by_commas>)`.
        4. The function `get_encoder_from_str` supports providing multiple encoders by using a single string containing valid encoder representations separated by commas.

    - **Build encoder objects from metadata**:
        1. Use the function `build_encoders_dict_from_metadata` from the `encode_factory.py` file.
        2. This function creates a list of encoders for each column based on the metadata.

4. **Fit and Normalize**: Use the `fit_and_normalize` function to calculate normalization parameters from the given data and normalize the data according to these parameters. Some encoders do not support normalization (e.g., boolean encoders, cyclic encoders). For these encoders, normalization is disabled by setting `self.fit_and_normalize` to `False`. Additionally, these encoders do not allow the provision of mean and standard deviation in their constructor.

5. **Normalize**: Use the `normalize` function to normalize all the columns in the given DataFrame according to the normalization parameters specified in the column names.



### scans_raw_parser:
1. Provides functions to:
    - Get all attributes and their values as a DataFrame.
    - Retrieve all evidence names.
    - Retrieve all decision names.
    - Filter attributes, evidences, and decisions using a list of valid values.

2. Includes a generic function named `extract_value_from_json` to parse a JSON representation and extract values for any specified key, regardless of the structure or depth.
