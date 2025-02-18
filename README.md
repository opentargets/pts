# PTS — Open Targets Pipeline Transformation Stage

Convert files into formats and structures used by the Open Targets data pipeline.


## Summary

This application uses the [Otter](http://github.com/opentargets/otter) library to
convert some input files into parquet, and transform some ontologies we use into
structures suitable for the Open Targets pipeline.

Check out the [config.yaml](config.yaml) file to see the steps and the tasks that
make them up.


## Installation and running

PTS uses [UV](https://docs.astral.sh/uv/) as its package manager. It is compatible
with PIP, so you can also fall back to it if you feel more comfortable.


```bash
uv run pts -h
```

> [!TIP]
> You can also use PTS with [Make](https://www.gnu.org/software/make/). Running
> `make` without any target shows help.


### Running with Docker

You can also launch PTS with Docker:
```bash
docker run ghcr.io/opentargets/pts:latest -h
```

PTS can upload the files it fetches into different cloud storage services. Open
Targets uses Google Cloud. To enable it in a docker container, you must have a
credentials file. Assuming you do, you can run the following command:

```bash
docker run \
  -v /path/to/credentials.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  ghcr.io/opentargets/pts:latest -h
```

To build your own Docker image, run the following command from the root of the
repository:

```bash
docker build -t pts .
```


## Development

> [!IMPORTANT]
> Remember to run `make dev` before starting development. This will set up a very
> simple git hook that does a few checks before committing.

Development of PTS can be done straight away in the local environment. You can run
the application just like before (`uv run pts`) to check the changes you make.

> [!TIP]
> Take a look at the [Otter docs](https://opentargets.github.io/otter), it is a
> very helpful guide when developing new tasks.

You can test the changes by running a small step, like `so`:

```bash
uv run pts --step so
```


## Copyright

Copyright 2014-2024 EMBL - European Bioinformatics Institute, Genentech, GSK,
MSD, Pfizer, Sanofi and Wellcome Sanger Institute

This software was developed as part of the Open Targets project. For more
information please see: http://www.opentargets.org

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
