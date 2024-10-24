# Graph Learning

There are two types of graph learning here.

1. [`graph-learning-iid.ipynb`](./graph-learning-iid.ipynb) treats every graph (CP)
   coming from an iid.
2. [`graph-learning-humemai.ipynb`](./graph-learning-humemai.ipynb) treats every graph
   (CP) to be an episodic memory of the agent. `pip install humemai==2.0.2`

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

## [`graph-learning-iid.ipynb`](./graph-learning-iid.ipynb)

- The raw data is converted into rdf-data, and is saved at
  [`./rdf-data-iid`](./rdf-data-iid)
- I also visualize the graph, which is saved at
  [`./graphs-visualized-iid`](./graphs-visualized-iid/)
- This method aims to find the N most general CPs (graphs) among the 211 of them.

## [`graph-learning-humemai.ipynb`](./graph-learning-humemai.ipynb)

- Every CP is treated as an "event", which is a collection of episode memories in
  HumemAI
- Currently saving them as episodic memories is done
- Now I have to do some kinda semantic memory extraction