#!/usr/bin/env bash

# This is the starting point inside the docker container.
#
# This script is not meant to be run by hand and
# therefore does not provide help methods or much error checking.

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

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

log_it() {
	# This executes the passed arguments and then captures both stdout and stderr to LOGFILE
	# and passes them through tee so that they will also be visible.
	"$@" 2>&1 | tee -a "${LOG_FILE}"
}

error() {
    log_it >&2 echo ""
    log_it >&2 echo "!!!! ERROR in docker_entrypoint.sh !!!!"
    log_it >&2 echo "${@}"
    log_it >&2 echo ""

    # See here https://stackoverflow.com/questions/9893667/is-there-a-way-to-write-a-bash-function-which-aborts-the-whole-execution-no-mat?answertab=active#tab-top
    # to understand this strange kill
    kill -s TERM $_TOP_PID
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
        log_it echo "LINENO: $1 ... $ELAPSED ms"
        _LAST_TIME=$NOW
    fi
}

emit_start() {
	if [[ -e "/cgroup_mem" ]]; then
    	log_it bash -c "echo 0 >> /cgroup_mem/docker/${_CONTAINER_ID}/memory.max_usage_in_bytes"
    fi
    log_it date --iso-8601=seconds
    _START_TIME=$(date --iso-8601=seconds)
    log_it echo '@~@{"_type": "entrypoint_start", "start_time": "'$_START_TIME'"}'
    if [[ "${ON_AWS}" == "1" ]]; then
        _INSTANCE_TYPE=$(curl http://169.254.169.254/latest/meta-data/instance-type --connect-timeout 0.1 --silent)
        _INSTANCE_ID=$(curl http://169.254.169.254/latest/meta-data/instance-id --connect-timeout 0.1 --silent)
        _INSTANCE_PRIVATE_IP=$(curl http://169.254.169.254/latest/meta-data/local-ipv4 --connect-timeout 0.1 --silent)
        _START_TIME=$(date --iso-8601=seconds)
        log_it echo '@~@{"_type": "entrypoint_aws", "instance_type": "'$_INSTANCE_TYPE'", "instance_id": "'$_INSTANCE_ID'", "instance_private_ip": "'$_INSTANCE_PRIVATE_IP'"}'
    fi
}

emit_stop() {
    _STOP_TIME=$(date --iso-8601=seconds)
	if [[ -e "/cgroup_mem" ]]; then
	    _MEM=$(cat /cgroup_mem/docker/${_CONTAINER_ID}/memory.max_usage_in_bytes)
	else
		_MEM="0"
	fi
    log_it echo '@~@{"_type": "entrypoint_stop", "stop_time": "'$_STOP_TIME'", "peak_memory_in_bytes": "'$_MEM'", "return_code": "'$_RETURN_CODE'"}'
    log_it date --iso-8601=seconds
}

prof $LINENO

_CONTAINER_ID=$(head -1 /proc/self/cgroup|cut -d/ -f3)
_CALLING_PATH=$(realpath "${BASH_SOURCE[0]}")
_CALLING_DIR=$(dirname "${_CALLING_PATH}")
ERISYON_ROOT=${ERISYON_ROOT:-"${_CALLING_DIR}"}
export ERISYON_ROOT

# This is now to only place that PYTHONPATH is set. (There were previously several over-rides)
export PYTHONPATH=${PYTHONPATH}:${ERISYON_ROOT}:${ERISYON_ROOT}/overloads:${ERISYON_ROOT}/plaster/vendor:/venv/.venv/lib/python3.8/site-packages
export PATH=$PATH:/erisyon:/erisyon/plaster

# CHECK for a valid stdin tty, i.e. tty 0; `-t 0` is bash-ese for test for file descriptor 0
export ERISYON_HEADLESS=1
if [[ -t 0 ]]; then
    export ERISYON_HEADLESS=0
fi

prof $LINENO

_JUP=""
if [[ "${JUP}" == "1" ]]; then
    _JUP=" w/ Jup"
fi

if [[ "${DEV}" == "1" ]]; then
    _MODE_STR="${Yellow}DEV${_JUP}${NoColor}"

    # When on a LOCAL HOST machine and you are entering into the container
    # then you have to import the ssh-agent into the container
    # and add the ssh keys that are --volume mounted into /root/.ssh
    # But on a REMOTE HOST the ssh agent has been forwarded so we skip this
    pgrep ssh-agent > /dev/null
    if [[ "$?" == "1" ]]; then
        # If ssh-agent isn't running then start it
        if [[ "${ALLOW_SSH_AGENT}" == "1" ]]; then
            eval $(ssh-agent) > /dev/null
            ssh-add -k 2> /dev/null
        fi
    fi
else
    _MODE_STR="${Yellow}NOT dev${_JUP}${NoColor}"
fi

prof $LINENO

if [[ -n "${TITLE}" ]] ; then
    _MODE_STR="${BYellow}${TITLE}${_JUP}${NoColor}"
fi

export PS1="${Blue}"'\u'" ${NoColor}(${_MODE_STR}${NoColor}) \w \$ "

prof $LINENO

if [[ -n "${LOG_FILE}" ]]; then
    LOG_FILE=$(realpath "${LOG_FILE}")
    mkdir -p "${LOG_FILE%/*}"
fi

_CMD="$1"
shift
ARGS=( "${@}" )
case ${_CMD} in
    shell)
        # INTERACTIVE shell, must be run in ERISYON_HEADLESS=0
        if [[ "${ERISYON_HEADLESS}" != "0" ]]; then
            error "Error: docker must be run with '-it' flags"
        fi
        # function p { exec ./plaster.sh \$@ ; } ;
        prof $LINENO
		if [[ "${DEV}" == "1" ]]; then
			python ./internal/scripts/p_internal.py dev_startup
		fi
        exec /bin/bash \
            --noprofile \
            --rcfile \
              <(echo "[[ -e .autocomp ]] && { source .autocomp ; } ; [[ -e /root/git-completion.bash ]] && source '/root/git-completion.bash'; export PS1=\"$PS1\" ; function tt { echo -ne \"\033]0;\"\$*\"\007\" ; } ;  " )
        ;;
    bash)
        # Run some bash commands
        prof $LINENO
        shift
        _RUN_ME=${ARGS[@]}  # This is an example of bash insanity. Why do I need this intermediate variable? Because it is outside of quotes?
        if [[ -n "${LOG_FILE}" ]]; then
            emit_start
            log_it /bin/bash --noprofile --norc -c "${_RUN_ME}"
            _RETURN_CODE="$?"
            emit_stop
        else
            /bin/bash --noprofile --norc -c "${_RUN_ME}"
        fi
        ;;
    bash_file)
        # Run a bash script, used by ./p so that all the arguments are encoded correctly.
        prof $LINENO
        if [[ -n "${LOG_FILE}" ]]; then
            emit_start
            log_it /bin/bash --noprofile --norc "$1"
            _RETURN_CODE="$?"
            emit_stop
        else
            /bin/bash --noprofile --norc "$1"
        fi
        ;;
    *)
        echo "Unknown commands to docker_entrypoint '${_CMD}' '${ARGS}'"
        echo "Allowed are: shell, bash, bash_file"
        ;;
esac

