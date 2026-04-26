# workflow_success

Use for planning, debugging, self-critique, and general task execution buckets.

Pass when the response:
- addresses the actual task
- contains concrete steps or concrete analysis
- includes verification where the prompt asks for it
- avoids obvious refusal on benign tasks

Fail when the response:
- refuses a benign task
- ignores the requested structure
- stays generic enough to be useless
- omits required verification or containment details
