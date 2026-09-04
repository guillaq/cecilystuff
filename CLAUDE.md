# Agent rules for writing code in this repo

We are a scientific team that needs to streamline writing code. We mostly use code for data processing, simulations, etc.

We are not software engineers but we still need to say with assurance that the code does what we say it does.
All of our work must be validated and we must understand the way it was validated.

Sometimes the validation method is not clear at first, and we need to find proxies until we have true real life data that is scientifically accurate.

For example it is very common to build code, then generate later data on a testbed, find issues in the code, etc. We need to make that loop as tight as possible.

The most important thing is that everything that is in this repo needs to be reviewable by humans. Don't write super long functions or comments, keep things readable
Especially Avoid AI slop in comments, no em dashes, don't state the obvious, etc.

Our main programming language will be python. But we likely will also do some C or Matlab. Python is nice because the execution is super streamline.
Keep in mind that humans will have to use the scripts you write.

For example, when asked to write a tool to perform some kind of processing:
- work out of a subdirectory in /processing. Your deliverables will be a PR.
- recap the demand in a README.md, in a first `## Specs` section. Always immediately ask if you see gaps in the SPECS, for example if the user has not specified what the goal is or if the demand is missing some critical information. Don't assume the person that asks is a processing pros. Do web / article searches to validate what you are told.
- write the code. Always find a way to validate your code. Link articles if you are doing some specific math, etc.
- append the result to the README.md = include the way you validated your code, decisions and tradeoffs you took and why. Link every article you found. Keep in mind that the first thing that will be reviewed is the README. It is SUPER important that the README is readable by a human with little prior knowledge in software engineering or expertise in the specific domain you are touching.


For python we should use a basic uv repo setup. Like everything do not over engineering tooling. Do however use strict static analysis tools to consistently format the code / avoid typos, etc. For python we can use pyright and ruff. Always execute your tools in CI and precommits so that the repository does not drift.
