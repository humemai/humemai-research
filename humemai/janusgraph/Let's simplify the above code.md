Let's simplify the above code. Below is the absolute necessary:

1. We want to add and delete memories. A memory is defined as a RDF-Star quadruple, a
   main triple with qualifier key-value pairs. So when we count the number of memories,
   one memory should include a main triple and all the qualifier key-value pairs added,
   not just one. Let's first start with this.

- `humemai.memoryID` is added as a qualifier
- `humemai.numRecalled` should also be added. This tells us how many times this memory
  was recalled. It's initialized with 0 when the memory is first added. Later on we'll
  add some recall functions, which will increment this value. So let's also make a
  function that finds the memory by memory id and then increments this value.

- Let's also make a method that deletes all the memories in the database.

2. We want to add / delete short-term memories. The short-term memories have to have
   some qualifiers.

- `humemai.currentTime` is a must qualifier. We are gonna use `datatype=XSD.dateTime`
  for this. If a current time is not specified when adding a short-term memory, then we
  will use the current time with `datetime.now().isoformat()`
- adding a short-term memory also take additional optional argument, which is
  `humemai.location`, which is a string value of the location.
- Remember that the basic add memory function adds `humemai.memoryID` and
  `humemai.numRecalled`. It shouldn't interfere with them.
- We want to add a method that deletes all the short-term memories. This is using SPARQL
  to find out the memories that have qualifier `currentTime` and removing all of them.

1. We want to add / delete long-term memories.

- Episodic memory is added with qualifiers. `humemai.eventTime` is a must qualifier. We
  are gonna use `datatype=XSD.dateTime`. It also has three optional qualifiers such as
  `humemai.location`, `humemai.emotion` (string) `humemai.event` (string).
- Semantic memory is added with two must qualifiers `humemai.derivedFrom` and
  `humemai.knownSince` (`datatype=XSD.dateTime`).
- Remember that the basic add memory function adds `humemai.memoryID` and
  `humemai.numRecalled`. It shouldn't interfere with them.
- We want to add a method that deletes all the long-term memories. This is using SPARQL
  to find out the memories that do not have qualifier `currentTime` and removing all of
  them.

4. We want to work on event. Some episodic memories have "events".
-  We want to have a method that retrieves all the memories by the name of the event.
-  Let's also have a mehtod that retrieves all the names of the events.

5. Working memory is defined as all short-term memories + partial long-term memory
- This method takes three arguments: `trigger_node`, `hops` and,
  `include_all_long_term`. If `include_all_long_term` is true then `trigger_node` and
  `hops` are not necessary, since we'll just return all the long-term memories
  along with the short ones. If `include_all_long_term` is False, then we have to
  consider both both `trigger_node` and `hops`, two of which determins the long-term
  memories that we want to fetch. We are gonna fetch all the memories that are within
  `hop`s from the `trigger_node`. There are several things to remember. We are not just
  fetching the nodes, but fetching the memories! Remember the definition of the memory
  we mentioned above. Also, we are considering both outgoing and incoming edges from the
  nodes in consideration.