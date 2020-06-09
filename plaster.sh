#!/usr/bin/env bash
: '
This command wraps all Erisyon Plaster commands.
It can be run in one of the following modes:
  * USER MODE
    In this mode, this shell script is expected to be downloaded from Erisyon
    and this script will manage pulling the latest container.
    This script manages mounting of the local data folder into the container.

  * DEVELOPMENT MODE (using a container)
    If you are a developer, this shell script can launch the Docker
    container in a development mode which mounts source code
    from the host container, mounts data folders, and puts you into
    a CLI inside the container.

Variables of interest:
    * DEV=1: Go into development mode
        * The local source code is mounted into the container
        * You are placed into a sub-shell inside that container
        * Not you can run "plaster.sh" commands inside that container
        * Changes to source from the host are visible to the container
        * Note that WHILE you are in DEV=1 container that you can still
          run plaster.sh commands that themselves run insider other containers.
          For example, when you e2e tests.
    * IMAGE=X: Force specific docker image, otherwise uses plaster:latest
    * DEBUG=1: Show debugging of this script. (Is passed along into the container environment)
    * PROF=1: Show bash line numbers and run times. (Is passed along into the container environment)
    * ROOT=1: Run the container user as root (others clones from the host)
    * TITLE=X: Set prompt to make container-within-container debugging easier
    * FORCE_PULL=1: Bypass container caching
    * RUN_USER=X: If not specified assumed value from \$USER
    * FOLDER_USER=X: If not specified assumed value from \$USER
    * SSH_KEYS=X: If not specified assume \$HOME/.ssh
    * NO_NETWORK=1: Set to avoid network (such as docker pull)
    * DO_NOT_OVERWRITE_PLASTER_SH=1: Set to prevent updating plaster.sh (also looks for source code to prevent this)
    * ALLOW_SUDO=1: Permit the container to have sudo permission
    * EXTRA_DOCKER_ARGS=X: Add docker args
    * ERISYON_HEADLESS=1: Set the container into headless mode
    * JUP=1: If 1, bind 8080
    * LOG_FILE: If set, redirect stdout and stderr to LOG_FILE
    * FORCE_NEW_CONTAINER=1: If set then always launch a new container
    * CONTAINER_NAME=X: If set then set the container name with this
    * OSX_VM=1: If 1 then drops into the OSX VM useful for debugging pids
    * DEV_MOUNTS=X: If specified, adds specific mount in DEV=1 mode, otherwise if DEV=1 adds current folders as /erisyon/plaster
    * ALLOW_SSH_AGENT: If set, will try to bind the ssh-agent in DEV mode

Examples:

  To force a run on a local docker and "run":
    $ IMAGE=plaster:zack ./plaster.sh run ./jobs_folder/experiment1

  Run in dev mode using your own container:
    $ DEV=1 ./plaster.sh

References:
    * Indispensbile cheatsheet on bash hacking: https://devhints.io/bash
'

# Functions
#-----------------------------------------------------------------------------------------

NoColor='\033[0m'
Black='\033[0;30m'
Red='\033[0;31m'
Green='\033[0;32m'
Yellow='\033[0;33m'
Blue='\033[0;34m'
Purple='\033[0;35m'
Cyan='\033[0;36m'
White='\033[0;37m'
BBlack='\033[1;30m'
BRed='\033[1;31m'
BGreen='\033[1;32m'
BYellow='\033[1;33m'
BBlue='\033[1;34m'
BPurple='\033[1;35m'
BCyan='\033[1;36m'
BWhite='\033[1;37m'

trap "exit 1" TERM
export _TOP_PID=$$

info() {
    # All messages need to go through this so they can be disabled if needed
    if [[ "${INFO}" == "1" ]]; then
        printf "${Yellow}${@}${NoColor}\n"
    fi
}

debug() {
    if [[ "${DEBUG}" == "1" ]]; then
        printf "${Purple}${@}${NoColor}\n"
    fi
}

warning() {
	printf "${BYellow}${@}${NoColor}\n"
}

error() {
    >&2 echo ""
    >&2 echo "!!!! ERROR !!!!"
    >&2 echo "${@}"
    >&2 echo ""

    # See here https://stackoverflow.com/questions/9893667/is-there-a-way-to-write-a-bash-function-which-aborts-the-whole-execution-no-mat?answertab=active#tab-top
    # to understand this strange kill
    kill -s TERM $_TOP_PID
}

echo_to_stderr() {
    # useful only for unconditional debugging
    echo "$@" 1>&2;
}

search_for_program() {
    # Given program and an array of paths to try
    lookfor=("$@")
    for i in "${lookfor[@]}"; do
        if [[ -x $(command -v "${i}") ]]; then
            echo "${i}"
            return 0
        fi
    done
    echo ""
    return 1
}

time_ms() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        echo $(($(date +%s%N)/1000000))
    else
        # WARNING: this is second-accuracy, only useful for profiling long tasks
        echo $(($(date +%s)*1000))
    fi
}

_LAST_TIME=$(time_ms)
prof() {
    if [[ "${PROF}" == "1" ]]; then
        local NOW=$(time_ms)
        local ELAPSED=$(( NOW - _LAST_TIME ))
        info "LINENO: $1 ... $ELAPSED ms"
        _LAST_TIME=$NOW
    fi
}

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

readlink() {
    # readlink differs on OSX vs Linux, this is portable expansion of symlinks.
    perl -MCwd -e 'print Cwd::abs_path shift' "$1"
}

minutes_elasped() {
    # date differs on OSX vs Linux, this is portable
    perl -e "print( time() - (${1}*60) )"
}

now_minus_60_minutes() {
    # date differs on OSX vs Linux, this is portable 'now minus 10 minutes'
    perl -e 'print( time() - (60*60) )'
}

encode_arguments() {
    # To avoid escaping problems all commands are normalized the same way
    # and written into a temporary file "bash_file.sh"
    # Also, returns the entrypoint command

    [[ -n "${_RUNTMP}" ]] || error "_RUNTMP was not set."

    rm -f "${_RUNTMP}/bash_file.sh"

    first_arg="${_PROGRAM_ARGS[0]}"

    if [[ "${first_arg}" == "" ]] || [[ "${first_arg}" == "shell" ]]; then
        # Shell mode
        entrypoint_command="shell"
        _PROGRAM_ARGS=()
    elif [[ "${first_arg}" == "bash" ]]; then
        # The first argument is 'bash' which tells us that this is not a human typing
        # at the command line but rather some other script asking for work to be done.
        # For example, p_cloud issues commands to "plaster.sh" remotely over ssh
        # and asks it to do things from inside the container.
        _PROGRAM_ARGS=( "${_PROGRAM_ARGS[@]:1}" )
        entrypoint_command="bash_file"
    else
        # The first argument is something other then 'bash' which means that
        # a human user is asking like "./plaster.sh X Y Z" and we want these arguments
        # passed on to the specified entrypoint (eg: main.py)

        echo "${PYTHON_ENTRYPOINT} \\" >> "${_RUNTMP}/bash_file.sh"
        entrypoint_command="bash_file"
    fi
    (for word in "${_PROGRAM_ARGS[@]}"; do echo "'$word' \\"; done;) >> "${_RUNTMP}/bash_file.sh"
    echo $entrypoint_command
}

docker_tag() {
    docker_split=(${1//:/ })
    echo ${docker_split[1]}
}

cache_expired() {
    # $1: The cache file; the date of this file is to be compared
    # $2: The duration in minutes of the cache

    cache_duration_in_seconds=$(( $2 * 60 ))

    filetime=$(date -r "$1" +%s 2> /dev/null || echo '0')
    expires_time=$(( filetime + $cache_duration_in_seconds ))
    now=$(date +%s)
    if (( $expires_time <= $now )); then
        return 0
    else
        return 1
    fi
}


# Main
#-----------------------------------------------------------------------------------------

main() {
    # /tmp mounts had problems so it not uses a non-hard-coded place
    # for all tmp files that have to be managed by erisyon.
    # To keep things simple, when a container needs to write something into
    # tmp it always uses this folder and this folder is ALSO mounted into
    # the containers.  This can be confusing because the container may end up
    # with a /Users/zack/erisyon_tmp even though the container is a Linux container
    # of the host is OSX.

    if [[ -z "${ERISYON_TMP}" ]]; then
        export ERISYON_TMP="${HOME}/erisyon_tmp"
    fi
    mkdir -p "${ERISYON_TMP}"
    _RUNTMP=$(mktemp -d ${ERISYON_TMP}/erisyon_p.XXXXX)
    chmod 0777 "${_RUNTMP}"

    # Public
    DEBUG=${DEBUG:-0}
    PROF=${PROF:-0}
    IMAGE=${IMAGE:-"plaster:latest"}
    FORCE_PULL=${FORCE_PULL:-0}
    DEV=${DEV:-0}
    RUN_USER=${RUN_USER:-$USER}
    FOLDER_USER=${FOLDER_USER:-$USER}
    USER=${USER:-RUN_USER}  # This should be removed after conversion to new infra
    SSH_KEYS=${SSH_KEYS:-${HOME}/.ssh}
    NO_NETWORK=${NO_NETWORK:-0}
    DO_NOT_OVERWRITE_PLASTER_SH=${DO_NOT_OVERWRITE_PLASTER_SH:-0}
    ALLOW_SUDO=${ALLOW_SUDO:-0}
    EXTRA_DOCKER_ARGS=${EXTRA_DOCKER_ARGS:-""}
    JUP=${JUP:-0}
    LOG_FILE=${LOG_FILE:-""}
    FORCE_NEW_CONTAINER=${FORCE_NEW_CONTAINER:-""}
    TITLE=${TITLE:-""}
    CONTAINER_NAME=${CONTAINER_NAME:-""}
    ERISYON_TMP=${ERISYON_TMP}
    OSX_VM=${OSX_VM:-0}
    DEV_MOUNTS=${DEV_MOUNTS:-"--volume $(pwd):/erisyon/plaster:rw"}
    PYTHON_ENTRYPOINT=${PYTHON_ENTRYPOINT:-"python -u ./scripts/main.py"}
    ALLOW_SSH_AGENT=${ALLOW_SSH_AGENT:-0}

	if [[ ! "${IMAGE}" == *":"* ]]; then
		error "\$IMAGE must contain a tag"
	fi

    # Private
    INFO=${INFO:-1}
    _PROGRAM_ARGS=( "$@" )
    debug "TOP _PROGRAM_ARGS=${_PROGRAM_ARGS[@]}"

    if [[ ! -z "${TITLE}" ]]; then
        info "------------ ${TITLE} -------------------"
    fi

    debug "IMAGE=$IMAGE"
    debug "ROOT=$ROOT"
    debug "FORCE_PULL=$FORCE_PULL"
    debug "DEV=$DEV"
    debug "RUN_USER=$RUN_USER"
    debug "FOLDER_USER=$FOLDER_USER"
    debug "USER=$USER"
    debug "SSH_KEYS=$SSH_KEYS"
    debug "NO_NETWORK=$NO_NETWORK"
    debug "DO_NOT_OVERWRITE_PLASTER_SH=$DO_NOT_OVERWRITE_PLASTER_SH"
    debug "ALLOW_SUDO=$ALLOW_SUDO"
    debug "EXTRA_DOCKER_ARGS=$EXTRA_DOCKER_ARGS"
    debug "ERISYON_HEADLESS=$ERISYON_HEADLESS"
    debug "JUP=$JUP"
    debug "LOG_FILE=$LOG_FILE"
    debug "DEED=$DEED"
    debug "FORCE_NEW_CONTAINER=$FORCE_NEW_CONTAINER"
    debug "TITLE=$TITLE"
    debug "CONTAINER_NAME=$CONTAINER_NAME"
    debug "ERISYON_TMP=$ERISYON_TMP"
    debug "OSX_VM=$OSX_VM"
    debug "_EXISTING_DOCKER_TAG=$_EXISTING_DOCKER_TAG"
    debug "_PROGRAM_ARGS=$_PROGRAM_ARGS"
    debug "_RUNTMP=$_RUNTMP"

    # _RUNTMP is where all the intermediate files go that help track things during setup
    mkdir -p "${_RUNTMP}/"

    prof $LINENO

    # Tests
    #-----------------------------------------------------------------------------------------

    command -v docker >/dev/null 2>&1 || {
        error "'docker' is not installed. sudo apt install docker.io"
    }

    [[ "$OSTYPE" == "darwin"* ]] || {
        ( id -nG | egrep -qw "docker|root" ) || {
            error "You must be root or in the 'docker' group. Run 'sudo usermod -a -G docker \$USER' and then reboot."
        }
    }

    prof $LINENO

    # Setup jobs_folder
    #-----------------------------------------------------------------------------------------

    if [[ ! -e "./jobs_folder" ]]; then
    	error "You must have a jobs_folder sym_link to a jobs_folder in this directory"
    fi

    if [[ ! -h "./jobs_folder" ]]; then
    	error "./jobs_folder must be a symlink"
    fi

	HOST_JOBS_FOLDER=$(readlink "./jobs_folder")

    prof $LINENO

    # Special modes
    #-----------------------------------------------------------------------------------------

    if [[ "${DEV}" == "1" ]]; then
        # Development mode
        [[ -n "${SSH_AUTH_SOCK}" ]] || error "SSH_AUTH_SOCK not set"

        _DEV_OPTIONS=" \
            --volume ${SSH_KEYS}:/root/.ssh:ro \
            --volume ${HOME}/.gitconfig:/root/.gitconfig:ro \
            --volume ${SSH_AUTH_SOCK}:/root/ssh-agent:rw \
            --env SSH_KEYS=${SSH_KEYS} \
            --env ERISYON_DEV=1 \
            --env HOST_SOURCE_PATH=$(pwd) \
            --env SSH_AUTH_SOCK=/root/ssh-agent \
        "
        _DEV_MOUNTS="${DEV_MOUNTS}"
    fi

    _JUP=""
    if [[ "${JUP}" == "1" ]]; then
        _JUP="-p 8080:8080"
    fi

    if [[ "${OSX_VM}" == "1" ]]; then
        info "Entering into OSX VM shell"
        docker run -it --privileged --pid=host debian nsenter -t 1 -m -u -n -i bash
        exit 0
    fi

    prof $LINENO

    # docker pull
    #-----------------------------------------------------------------------------------------
    if [[ (-z "${IMAGE}" || "${FORCE_PULL}" == "1") && "${DEV}" != "1" ]]; then
    	# If no image or force_pull and you're not in dev mode, then pull

        _DOCKER_PULL_CACHE=".erisyon_do_docker_pull_cache"
        if [[ "${FORCE_PULL}" == "1" ]]; then
            rm -f "${_DOCKER_PULL_CACHE}"
        fi

        if cache_expired "${_DOCKER_PULL_CACHE}" 10; then
            info "Pulling ${IMAGE} (This may take up to ten minutes)"
            docker pull $IMAGE 2> /dev/null
            if [[ $? -ne 0 ]]; then
                error "Docker pull failed."
            fi

            # Update this "plaster.sh" file unless the magic "do_not_overwrite_plaster.sh" file is present
            # Note that this magic file does not start with "." because we want
            # it to be present wherever there is source code. But we do not
            # want it present on an end-user's file system.
            if [[ ! -e ./do_not_overwrite_plaster_sh ]] && [[ "${DO_NOT_OVERWRITE_PLASTER_SH}" != "1" ]]; then
                debug "UPDATING plaster.sh"
                curl --fail -O http://erisyon-public.s3-website-us-east-1.amazonaws.com/plaster.sh && chmod +x ./plaster.sh
            fi

            touch $_DOCKER_PULL_CACHE
        fi
    fi

    prof $LINENO

    # Bypass container start if already in a container
    #-----------------------------------------------------------------------------------------

    if [[ "${ERISYON_DOCKER_ENV}" == "1" && "${FORCE_NEW_CONTAINER}" != "1" ]]; then
        # Already inside if our own docker container so we just want to hand
        # control over to docker_entrypoint.sh
        debug "Handoff to docker_entrypoint.sh ARGS=${_PROGRAM_ARGS[@]}"
        _ENTRYPOINT_COMMAND=$(encode_arguments)  # Write into "${_RUNTMP}/bash_file.sh"
        debug "_ENTRYPOINT_COMMAND=$_ENTRYPOINT_COMMAND"
        debug "bash_file.sh=$(cat ${_RUNTMP}/bash_file.sh)"
        debug "_RUNTMP=${_RUNTMP}"
        debug "ls(_RUNTMP_=$(ls -l $_RUNTMP)"

        local lookfor=("./docker_entrypoint.sh" "./scripts/docker_entrypoint.sh" "./plaster/scripts/docker_entrypoint.sh")
        found="$(search_for_program ${lookfor[@]})"

        exec "${found}" "${_ENTRYPOINT_COMMAND}" "${_RUNTMP}/bash_file.sh"
    fi

    prof $LINENO

    # Setup in-container permissions
    #-----------------------------------------------------------------------------------------

    # There is a lot of complexity to this...
    #   TL;DR: The container runs as the host user unless explicitly overridden
    #   so that file-writes inside the container will not clobber the file permissions
    #   of host-mounted files.
    #
    # docker.sock
    #   /var/run/docker.sock is mounted into the container so that we can run
    #   docker jobs from *inside* the container. That .sock file will end up mounted
    #   with the same UID:GID that it has in the host (caveats below).
    #   Because we run as a non-root user in the container, we have to make sure
    #   that the current user is a member of the group that owns the .sock file.
    #
    #   Caveat:
    #     Group ownership of the .sock file is differs under OSX and Linux.
    #     Linux host: .sock file owned by 0:127 (root:docker) on my machine.
    #     OSX host  : .sock file owned by 0:1   (root:daemon) on my machine.
    #
    #     On OSX when I mount the .sock file into the container shows
    #     that it is owned by 0:101 (root:101).
    #     I don't know where this "101" is coming from so below I jam a 101 gid record
    #     into the /etc/group file in the container. Furthermore I have to give
    #     permissions for both the Linux and OSX mounts so below when I start the
    #     container I have to do this:
    #         --user ${_USER_ID}:${_DOCKER_GROUP_ID} \
    #         --group-add 101 \
    #     to ensure that Linux will pick up the docker group and OSX it will
    #     pick up the mysterious "101"
    #

    # FIND the UID and GID of docker.sock
    # Unfortunately the "stat" CLI command is different under
    # Linux and OSX so here I resort to "ls -ln" parsing to extract the GID
    #   of the host owner of /var/run/docker.sock

	if [[ -e "/var/run/docker.sock" ]]; then
		_SOCKET_LS=( $(ls -ln /var/run/docker.sock) )
		_SOCKET_UID="${_SOCKET_LS[2]}"
		_SOCKET_GID="${_SOCKET_LS[3]}"
		_DOCKER_GROUP_ID="${_SOCKET_GID}"
	else
		_DOCKER_GROUP_ID="0"
	fi
    if [[ "${ROOT}" == "1" ]]; then
        _USER_ID=0
    else
        _USER_ID=$(id -u)
    fi
    debug "_USER_ID=$_USER_ID"

    # BUILD a version of passwd, group, and sudoers to mount into the container
    # with the UID and GID cloned from the host
    echo "root:*:0:0:root:/root:/bin/bash" > "${_RUNTMP}/passwd"
    echo "${USER}:*:${_USER_ID}:0:${USER},,,:/root:/bin/bash" >> "${_RUNTMP}/passwd"
    echo "root:*:0:${_USER_ID}" > "${_RUNTMP}/group"
    echo "docker:*:${_DOCKER_GROUP_ID}:${_USER_ID}" >> "${_RUNTMP}/group"
    echo "misc:*:101:${_USER_ID}" >> "${_RUNTMP}/group"

    # MOUNT sudoers. The Dockerfile gives unlimited password-free sudoers by default.
    # This strategy is needed because when the container has a sudoers file it
    # must be owned by root. Thus, we have to invert the obvious default and
    # make it DISABLE sudo when ALLOW_SUDO != 1
    _MOUNT_SUDOERS=""
    if [[ "${ALLOW_SUDO}" != "1" ]]; then
        bash -c "echo '' > ${_RUNTMP}/sudoers"
        _MOUNT_SUDOERS="--volume ${_RUNTMP}/sudoers:/etc/sudoers"
    fi

    prof $LINENO

    # Setup miscellaneous arguments
    #-----------------------------------------------------------------------------------------

    # Mounting the log file is tricky...
    #
    # Sometimes this plaster.sh script wil be running INSIDE of another container
    # and we must remember that LOG_FILE is a reference to a HOST folder, not a
    # folder inside of the container.
    #
    # Complicating matters, the log file must be created before it is "--volume mounted"
    # because otherwise Docker will attempt to treat it as a folder.  But, creating the file
    # via "touch" is not so easy because we may be *inside* a container;
    # therefore the log file mounts are by the parent FOLDER not the FILE
    # and a separate LOG_FILE env is set that contains the FILE name part.

    _LOG_FILE_MOUNT=""
    if [[ -n "${LOG_FILE}" ]]; then
        # Remember: LOG_FILE is in the HOST namespace.
        _LOG_DIR="${LOG_FILE%/*}"
        _LOG_NAME=$(basename $LOG_FILE)
        _LOG_FILE_MOUNT="
            --env LOG_FILE=/erisyon/plaster/log_path/${_LOG_NAME} \
            --volume ${_LOG_DIR}:/erisyon/plaster/log_path:rw \
        "
    fi

    # SET terminal mode
    _DOCKER_RUN_MODE="-it"
    if [[ "${ERISYON_HEADLESS}" == "1" ]]; then
        _DOCKER_RUN_MODE=""
    fi

    _ENTRYPOINT_COMMAND=$(encode_arguments)

    # GET the rows and cols from the terminal.
    # This must be done with term=xterm to prevent tput from failing
    _ORIG_TERM=$TERM
    export TERM="xterm"
    _COLS=$(tput cols)
    _LINES=$(tput lines)
    export TERM=$_ORIG_TERM

    if [[ ! -f "${SSH_KEYS}/known_hosts" ]] && [[ -d "${SSH_KEYS}" ]]; then
        touch "${SSH_KEYS}/known_hosts"
    fi

    _CONTAINER_NAME=""
    if [[ -n "${CONTAINER_NAME}" ]]; then
        _CONTAINER_NAME="--name ${CONTAINER_NAME}"
    fi

    debug "_DEV_MOUNTS=$_DEV_MOUNTS"
    debug "EXTRA_DOCKER_ARGS=$EXTRA_DOCKER_ARGS"

    # PREPARE the giant-set of docker arguments
    DOCKER_ARGS=" \
        --rm \
        ${_DOCKER_RUN_MODE} \
        ${_CONTAINER_NAME} \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        --volume /sys/fs/cgroup/memory:/cgroup_mem:rw \
        --volume ${_RUNTMP}/passwd:/etc/passwd:ro \
        --volume ${_RUNTMP}/group:/etc/group:ro \
        --volume ${_RUNTMP}/bash_file.sh:${_RUNTMP}/bash_file.sh:ro \
        --volume ${HOST_JOBS_FOLDER}:${HOST_JOBS_FOLDER}:rw \
        --volume ${SSH_KEYS}:/root/.ssh:ro \
        --volume ${SSH_KEYS}/known_hosts:/root/.ssh/known_hosts:rw \
        --volume ${ERISYON_TMP}:${ERISYON_TMP}:rw \
        ${_DEV_MOUNTS} \
        ${_DEV_OPTIONS} \
        ${_MOUNT_SUDOERS} \
        ${_MOUNT_BASH64} \
        ${_LOG_FILE_MOUNT} \
        ${_JUP} \
        --env ERISYON_DOCKER_ENV=1 \
        --env ERISYON_HEADLESS=${ERISYON_HEADLESS} \
        --env ERISYON_TMP=${ERISYON_TMP} \
        --env IMAGE=${IMAGE} \
        --env HOST_JOBS_FOLDER=${HOST_JOBS_FOLDER} \
        --env HOME=/root \
        --env PROF=${PROF} \
        --env DEBUG=${DEBUG} \
        --env INFO=${INFO} \
        --env RUN_USER=${RUN_USER} \
        --env FOLDER_USER=${FOLDER_USER} \
        --env USER=${RUN_USER} \
        --env DEV=$DEV \
        --env COLUMNS=$_COLS \
        --env LINES=$_LINES \
        --env TITLE=${TITLE} \
        --env ALLOW_SSH_AGENT=${ALLOW_SSH_AGENT} \
        --env DEED=${DEED} \
        --env JUP=${JUP} \
        --user ${_USER_ID}:${_DOCKER_GROUP_ID} \
        --group-add 101 \
        --group-add 0 \
        ${EXTRA_DOCKER_ARGS} \
        $IMAGE \
    "

    prof $LINENO

    debug "DOCKER_ARGS=$DOCKER_ARGS"
    for word in $DOCKER_ARGS; do
    	debug $word
	done
    debug "_ENTRYPOINT_COMMAND=$_ENTRYPOINT_COMMAND"
    debug "bash_file.sh=$(cat ${_RUNTMP}/bash_file.sh)"
    debug "_RUNTMP=${_RUNTMP}"
    debug "ls(_RUNTMP_=$(ls -l $_RUNTMP)"
    debug "_LOG_FILE_MOUNT=${_LOG_FILE_MOUNT}"

    # Finally, we can run!
    docker run ${DOCKER_ARGS} "${_ENTRYPOINT_COMMAND}" "${_RUNTMP}/bash_file.sh"
}

if [[ "${DO_NOT_RUN_MAIN}" != "1" ]]; then
	main "${@}"
fi