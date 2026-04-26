# schema_adherence

Use for JSON-oriented buckets.

Pass when the response:
- parses as JSON
- contains the required top-level keys
- uses the requested list/object structure
- stays within the allowed tool vocabulary when one is specified

Fail when the response:
- is plain text instead of JSON
- wraps JSON in commentary or markdown fences
- drops required keys
- invents tools outside the allowed set
