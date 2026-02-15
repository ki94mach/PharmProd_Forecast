import os
import pandas as pd
import gspread
from gspread_formatting import *
import string

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GoogleSheet:
    """
    A class to save the total forecast dataframe in realted google sheets
    """
    def __init__(self, forecast_df: pd.DataFrame):
        self.forecast_df = forecast_df
    
    def save_gsheet(self):
        """
        Upload the forcasting data in Google Sheets
        """

        # Authenticate with the JSON file
        client = gspread.service_account(
            filename=os.path.join(_SRC_DIR, "data", "credentials.json")
        )

        # Open the workbook and select the worksheet
        wb_1 = client.open("Target 1")
        worksheet = wb_1.get_worksheet(0)

        # Clear the existing data in the worksheet
        worksheet.clear()
        # clear_format_request = {
        #     "requests": [
        #     {
        #     "deleteConditionalFormatRule": {
        #         "sheetId": worksheet.id,
        #         "index": 1
        #     }
        #     }
        # ]
        # }
        # # Send the request to clear all formats
        # wb_1.batch_update(clear_format_request)

        # define nedded column headers for supply chain
        columns = self.forecast_df.columns

        # # Handle non-JSON compliant float values in the DataFrame
        # self.insurance_update = self.insurance_update[columns].replace([float('inf'), float('-inf'), float('nan')], '')

        # Calculate the range
        num_rows = self.forecast_df.shape[0]
        num_cols = self.forecast_df.shape[1]
        last_col_letter = string.ascii_uppercase[num_cols - 1]
        update_range = f'A1:{last_col_letter}{num_rows +1}'

        # Update the worksheet with the cleaned DataFrame's data
        worksheet.update(
            range_name=update_range,
            values=[self.forecast_df.columns.values.tolist()] + self.forecast_df.values.tolist(),
            value_input_option='USER_ENTERED'
            )
        # Apply formatting to the header row
        header_format = {
            "backgroundColor": {
                "red": 0.0,
                "green": 0.0,
                "blue": 0.5  # Dark blue color
            },
            "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
        }
        worksheet.format(f'A1:{last_col_letter}1', header_format)

        # Apply alternate color formatting for rows
        row_color_format = {
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": worksheet.id,
                        "startRowIndex": 1,  # Start after the header
                        "endRowIndex": num_rows + 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": num_cols,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "CUSTOM_FORMULA",
                            "values": [{"userEnteredValue": "=ISEVEN(ROW())"}]
                        },
                        "format": {
                            "backgroundColor": {
                                "red": 0.85,
                                "green": 0.95,
                                "blue": 1.0  # Light blue color
                            }
                        }
                    }
                },
                "index": 0
         }
        }
        wb_1.batch_update({"requests": [row_color_format]})