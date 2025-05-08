# deck reshuffling

add possibility for the user to pick locations for reagents

## needs

- default deck locations model
  - pydantic grid based | expand layout class with named locations
- show current deck
  - grid based layout (A1, A2, B1, B2)
  - labware types
- use default deck locations as a template
- expose movable deck defaults

# protocol runner

allow parsing protocols to extract deck before running

- rename Protocol class to CommandBuilder
- higher level Protocol class that can pass a deck and specific locations to the CommandBuilder
  - need to parse deck beforehand
  - return through the API
    - probably called on accessing the config page
    - add deck preview to the config page
    - instructions for placing labware when protocol is configured

show default layout -> config & compile -> show deck layout with instructions

# sleep in protocol

need to be able to wait/pause (example incubation) in a protocol
