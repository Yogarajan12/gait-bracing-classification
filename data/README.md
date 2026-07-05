# Data

The notebook expects a single CSV in long format, one row per
(subject, condition, replication, leg, joint, time) observation.

## Expected path

The notebook loads the file from Google Drive:

    data_path = '/content/drive/MyDrive/.../gait.csv'
    df = pd.read_csv(data_path)

When running locally, replace this with a path to your copy, for example
`data/gait.csv`, and remove the `google.colab` drive-mount cell.

## Schema

| Column        | Type  | Description                                            |
|---------------|-------|--------------------------------------------------------|
| subject       | int   | Subject identifier (10 subjects)                       |
| condition     | int   | 1 = unbraced, 2 = knee-braced, 3 = ankle-braced        |
| replication   | int   | Repeated trial index for a subject and condition       |
| leg           | int   | 1 = left, 2 = right                                    |
| joint         | int   | 1 = ankle, 2 = knee, 3 = hip                           |
| time          | int   | Gait-cycle percentage, 0 to 100 (101 points per cycle) |
| angle         | float | Joint angle at that time point                         |

The raw file has 181,800 rows, which reshape into a tensor of shape
(300, 101, 6): 300 cycles (100 per condition), 101 time points, and 6
joint-leg channels ordered as left ankle, left knee, left hip, right ankle,
right knee, right hip.

## Source

Add the dataset source and citation here before publishing, and confirm the
data license permits redistribution before committing the CSV itself.
