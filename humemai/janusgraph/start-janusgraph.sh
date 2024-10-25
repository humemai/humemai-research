#!/bin/bash

# Define paths for configuration files
JANUSGRAPH_CONFIG="./configs/janusgraph.properties"
GREMLIN_SERVER_CONFIG="./configs/gremlin-server.yaml"

# Ensure the configs directory exists
mkdir -p ./configs

# Check if janusgraph.properties exists; if not, create it
if [ ! -f "$JANUSGRAPH_CONFIG" ]; then
    echo "Creating janusgraph.properties..."
    cat > "$JANUSGRAPH_CONFIG" <<EOL
storage.backend=cassandra
storage.hostname=cassandra
storage.cassandra.keyspace=janusgraph
EOL
fi

# Check if gremlin-server.yaml exists; if not, create it
if [ ! -f "$GREMLIN_SERVER_CONFIG" ]; then
    echo "Creating gremlin-server.yaml..."
    cat > "$GREMLIN_SERVER_CONFIG" <<EOL
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
EOL
fi

# Run docker-compose
docker-compose up -d
