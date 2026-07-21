Summary: Example memory file showing the expected format.

This is an example long-term memory file for the porygon agent.

The format is simple: the first line is `Summary: <one line>` — that summary is
what the agent sees when it runs `memory list`, so keep it short and
descriptive. After a blank line, the rest of the file is a free-form markdown
body that the agent only loads when it decides to `read` this file by name.

You can create files like this by hand, or ask the agent to remember something
and it will write one via the `memory` tool (a `.md` suffix is added
automatically if you omit it).
