# Graph Learning


## [`process-raw-data`](./process-raw-data.ipynb)

- This jupyter notebook processes the raw data (saved at
  [`./user-raw-data/new`](./user-raw-data/new)) that Emma gave me.
- It processes them and saves the data as `raw-data.json`.

## [`raw-data.json`](./raw-data.json)

- There are in total of 211 CPs (elements in a list) saved.
- Below is an example CP (saved as a dict):

  ```json
  {
    "cp_num": 1,
    "participant": 4071,
    "cp_name": "stand still break big rock",
    "ticks_lasted": 844,
    "round_num": 1,
    "timestamp": "2024-08-29T15:15:10",
    "unix_timestamp": 1724937310,
    "remaining_time": 3000,
    "remaining_rocks": 23,
    "victim_harm": 1200,
    "situation": [
      [
        {
          "type": "object",
          "content": "Large rock"
        },
        {
          "type": "location",
          "content": "Top of rock pile"
        }
      ],
      [
        {
          "type": "object",
          "content": "Large rock"
        },
        {
          "type": "location",
          "content": "<Left> side of rock pile"
        }
      ],
      [
        {
          "type": "location",
          "content": "<Right> side of field"
        },
        {
          "type": "actor",
          "content": "Human"
        }
      ]
    ],
    "HumanAction": [
      [
        {
          "type": "action",
          "content": "Stand still in <location>"
        },
        {
          "type": "location",
          "content": "<Right> side of field"
        }
      ],
      [],
      []
    ],
    "RobotAction": [
      [],
      [
        {
          "type": "action",
          "content": "Break <object> in <location>"
        },
        {
          "type": "object",
          "content": "Large rock"
        },
        {
          "type": "location",
          "content": "Top of rock pile"
        }
      ],
      []
    ]
  }
  ```

## [`graph-learning-humemai.ipynb`](./graph-learning-humemai.ipynb)

- We use the python humemai package to store data
  - `pip install humemai==2.4.1`
- The data is stored in a graph database
- We try to find the most representative CP with graph learning