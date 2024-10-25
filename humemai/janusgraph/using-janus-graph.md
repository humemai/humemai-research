## Necessary config files

`gremlin-server.yaml`:

```
host: 0.0.0.0
port: 8182
scriptEvaluationTimeout: 30000
channelizer: org.apache.tinkerpop.gremlin.server.channel.WebSocketChannelizer
graphs: {
  graph: conf/janusgraph.properties
}

serializers:
  - { className: org.apache.tinkerpop.gremlin.driver.ser.GraphSONMessageSerializerV3d0,
      config: { 
          ioRegistries: [org.janusgraph.graphdb.tinkerpop.JanusGraphIoRegistry] 
      }
    }
  - { className: org.apache.tinkerpop.gremlin.driver.ser.GryoMessageSerializerV3d0 }

# Optional: Enable SSL if required
# ssl: true
# keyCertChainFile: conf/server.pem
# keyFile: conf/server.key
```

`janusgraph.properties`:

```
storage.backend=cassandra
storage.hostname=cassandra
storage.cassandra.keyspace=janusgraph
```

## Running docker 

### Running cassandra backend

Create a container if you haven't
```sh
docker run -d --name cassandra -p 9042:9042 cassandra
```

Start (resume) the container
```sh
docker start cassandra
```

### Running janusgraph 

Create a container
```sh
docker run -d --name janusgraph \                    
  --link cassandra:cassandra \
  -p 8182:8182 \
  -v ~/janusgraph-config/janusgraph.properties:/opt/janusgraph/conf/janusgraph.properties \
  -v ~/janusgraph-config/gremlin-server.yaml:/opt/janusgraph/conf/gremlin-server.yaml \
  janusgraph/janusgraph
```

Start (resume) the container
```sh
docker start janusgraph
```

## Running example python code

```python
import nest_asyncio
import asyncio
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.structure.graph import Graph

# Apply nest_asyncio to allow nested event loops (useful in Jupyter notebooks)
nest_asyncio.apply()

# Set up the connection using GraphSON 3.0
graph = Graph()
connection = DriverRemoteConnection(
    'ws://localhost:8182/gremlin',
    'g',
    message_serializer=GraphSONSerializersV3d0()
)
g = graph.traversal().withRemote(connection)

try:
    # Clear existing data (optional but useful for testing)
    g.V().drop().iterate()

    # Add Persons
    alice = g.addV('person').property('name', 'Alice').property('age', 30).next()
    bob = g.addV('person').property('name', 'Bob').property('age', 25).next()
    carol = g.addV('person').property('name', 'Carol').property('age', 27).next()
    dave = g.addV('person').property('name', 'Dave').property('age', 35).next()

    # Add Organizations
    acme = g.addV('organization').property('name', 'Acme Corp').property('type', 'Company').next()
    globex = g.addV('organization').property('name', 'Globex Inc').property('type', 'Company').next()

    # Add 'knows' relationships
    g.V(alice.id).addE('knows').to(bob).property('since', 2015).iterate()
    g.V(alice.id).addE('knows').to(carol).property('since', 2018).iterate()
    g.V(bob.id).addE('knows').to(dave).property('since', 2020).iterate()
    g.V(carol.id).addE('knows').to(dave).property('since', 2019).iterate()

    # Add 'works_at' relationships
    g.V(alice.id).addE('works_at').to(acme).property('role', 'Engineer').iterate()
    g.V(bob.id).addE('works_at').to(globex).property('role', 'Analyst').iterate()
    g.V(carol.id).addE('works_at').to(acme).property('role', 'Manager').iterate()
    g.V(dave.id).addE('works_at').to(globex).property('role', 'Director').iterate()

    print("Vertices and edges added successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the connection
    connection.close()
```

```python
import nest_asyncio
import asyncio
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.structure.graph import Graph

nest_asyncio.apply()

graph = Graph()
connection = DriverRemoteConnection(
    'ws://localhost:8182/gremlin',
    'g',
    message_serializer=GraphSONSerializersV3d0()
)
g = graph.traversal().withRemote(connection)

try:
    persons = g.V().hasLabel('person').valueMap().toList()
    for person in persons:
        print(person)
finally:
    connection.close()
```