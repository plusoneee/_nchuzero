## Google Colab tips
`Colab` is free platform from Google that allow us to code in Python.

### Load CSV files into colab
When loading files like `.csv`, it requires some extra coding.
* From GitHub
* From Local Drive
* Mount Drive

#### From GitHub
1. Click on the dataset in GitHub Repository, then click `View Raw`. 
2. Copy the link and store it as `url`.
3. To get the dataframe by load the url into pandas `read_csv()`. 
4. note: files < 25M.
```python
import pandas as pd
url = 'YOUR_RAW_GITHUB_LINK'
myDataframe = pd.read_csv(url)
```

#### From Local Drive
```python
from google.colab import files
uploaded = files.upload()
```
You should see the name of the file after it is uploaded. The `uploaded` variable is a `dict` type. Type in the following code to import it into dataframe.
```python
import io
datframe1 = pd.read_csv(io.BytesIO(uploaded['FILE_NAME.csv']))
# or
dataframe2 = pd.pd.read_csv('./FILE_NAME.csv')
```


#### Reference
1. [3 Ways to Load CSV files into Colab](https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92)