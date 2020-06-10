# Plaster: Erisyon's Fluorosequencing Informatic Pipeline

This is the repository for Erisyon's analysis software for analyzing and simulating
runs on our fluorosequecning platform.

It consists of the following parts:

1. gen: A CLI tool that generates instructions for plaster.
1. run: The tool that runs the analysis. It is comprised of the following parts:
    * Sigproc: The signal processor that reads raw images from the instrument
       and coverts that into a "radmat" (rad-iometry mat-rix).
    * Simulator aka Virtual Fluoro-Sequencing (VFS): Uses an error model of the chemistry
       and instrument to produce simulated reads by Monte Carlo sampling.
       Used to train and evaluate the classifier.
    * Classifier: Trained against simulated reads, produces peptide or protein calls from reads,
       either real or simulated from VFS.

# Getting Started

## Requirements

The following assumes you have OSX or Linux as the host environment.
If you are using Ubuntu Linux make sure you are on an LTS (.04) release.

1. "docker" installed.
    - Test: `sudo docker --version`
    - If not, Linux: `sudo apt install -y docker.io`; [OSX](https://docs.docker.com/docker-for-mac/install/)
1. "docker" needs to be runnable without sudo.
    - Test: `docker run hello-world`
    - If not: Linux: `sudo gpasswd -a $USER docker` and reboot; OSX: (Unknown, possibly the same?)
1. Pull source, canonically in ~/git/plaster, but you can put it where you need to.
   Henceforth, this readme will assume this path is in `$PLASTER_ROOT`
    - `git clone git@github.com:erisyon/plaster.git`

If you are running on a local machine:

```bash
$ cd $PLASTER_ROOT

# Create a symlink to where data will be stored
$ ln -s /a/folder/for/data ./jobs_folder

# Build the container
$ docker build -t plaster:latest .

# Start a shell inside that container.
$ ./plaster.sh
```

# Usage

1. Generate plaster instructions
    ```bash
    $ ./plaster.sh gen --help
    ```

1. Run plaster on those instructions
    ```bash
    $ ./plaster.sh run --help
    ```
   
## Example

VFS on ten random proteins from the Human Proteome with 4 labels.
Writes the "job" into `./jobs_folder/human_random_10`.

```bash
$ p gen classify \
  --job=human_random_10 \
  --sample='human_proteome_peptides_trypsin_edmans' \
  --protein_csv=s3://erisyon-public/reference_data/human_proteome_peptides_trypsin.csv \
  --protein_random=10 \
  --label_set='DE,Y,M,H' \
  --n_edmans=10 \
  --n_pres=1

$ p run ./jobs_folder/human_random_10
# Report output in ./jobs_folder/human_random_10/report.html
# or the notebook is in ./jobs_folder/human_random_10/report.ipynb
```

# Development

Source files are maintained on the host OS and but running tasks (tests, etc) is
done within a Docker container with the source files mounted into the container.

## Development in the container

```bash
$ cd $PLASTER_ROOT
$ DEV=1 ./plaster.sh
```

This mounts the host's source files into the container and drops you into a shell ready
to run further commands.
You can now edit files in your host's IDE and run commands in this shell.

Examples
```bash
$ ./plaster.sh --help
$ ./plaster.sh test  # Runs all tests
```

# Jupyter

In order to start a local Jupyter you need to enable the port from the host. So before you go into
dev mode: `JUP=1 DEV=1 ./plaster.sh jupyter` or `JUP=1 ./plaster.sh jupyter` to run with the latest container.
