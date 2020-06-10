#!/usr/bin/env bash

# This is the entrypoint for e2e tests which test environmental setup
# and are therefore run _outside_ of the docker container.

if [[ "${ERISYON_DOCKER_ENV}" == "1" ]]; then
	echo "Error: E2E test should be run outside of the docker environment"
	exit 1
fi

USE_DOCKER_CACHE=${USE_DOCKER_CACHE:-"1"}
_DOCKER_CACHE=""
if [[ "${USE_DOCKER_CACHE}" != "1" ]]; then
	_DOCKER_CACHE="--no-cache"
fi

set -euo pipefail
trap "exit 1" TERM
export _TOP_PID=$$
_error() {
    echo ""
    echo "!!!! ERROR in e2e.sh !!!!"
    echo "${@}"
    echo ""

    # See here https://stackoverflow.com/questions/9893667/is-there-a-way-to-write-a-bash-function-which-aborts-the-whole-execution-no-mat?answertab=active#tab-top
    # to understand this strange kill
    kill -s TERM $_TOP_PID
}

_run() {
	type -t "$1" | grep "function" > /dev/null
	if [[ "$?" == "0" ]]; then
		echo "Running : $1"
		$1
		return $?
	else
		echo "Skipping: $1"
		return 0
	fi
}

_check_plaster_root() {
	if [[ ! -e "plaster_root" ]]; then
		_error "Must be run from plaster_root"
	fi
}

_fail_test() {
	_error "TEST FAILED LINE ${1}"
}

_check_fail_test() {
	[[ "${1}" == "0" ]] || _fail_test "${2}"
}

_sleep() {
	echo "Sleeping: $1 seconds..."
	sleep $1
}

_check_plaster_root
_ERISYON_ROOT=$(pwd)



# plaster container tests
# ------------------------------------------------------------------------------

test_plaster_docker_container() {
	_check_plaster_root

	# Note: This is testing plaster build directly without docker_build.sh
	docker build ${_DOCKER_CACHE} -t "plaster:e2e" .

	it_passes_tests() {
		LIMIT_TESTS=""
		IMAGE=plaster:e2e ./plaster.sh test "${LIMIT_TESTS}" --no_clear
		_check_fail_test $? $LINENO
	}

	it_gens_and_runs() {
		rm -rf ./jobs_folder/__e2e_test
		IMAGE=plaster:e2e ./plaster.sh gen classify \
			--job=__e2e_test \
			--sample=e2e_test \
		    --decoys=none \
    		--protein_seq="pep25:GCAGCAGAG" \
    		--n_pres=0 \
		    --n_edmans=8 \
		    --label_set='C'
		_check_fail_test $? $LINENO

		IMAGE=plaster:e2e ./plaster.sh run ./jobs_folder/__e2e_test
		_check_fail_test $? $LINENO
	}

	it_starts_plaster_container_as_non_dev() {
		IMAGE=plaster:e2e ./plaster.sh gen --readme | grep -i "GEN -- The plaster run generator" > /dev/null
		_check_fail_test $? $LINENO

		echo 1 > __e2e_test_file
		IMAGE=plaster:e2e ./plaster.sh bash ls -l | grep -i "__e2e_test_file" > /dev/null
		[[ "$?" == "1" ]] || _check_fail_test $? $LINENO
		rm __e2e_test_file
	}

	it_starts_plaster_container_as_dev() {
		echo 1 > __e2e_test_file
		DEV=1 IMAGE=plaster:e2e ./plaster.sh bash ls -l | grep "__e2e_test_file" > /dev/null
		_check_fail_test $? $LINENO
		rm __e2e_test_file
	}

	it_runs_jupyter() {
		rm -f ./scripts/e2e_test_notebook.html
		DEV=1 IMAGE=plaster:e2e \
			./plaster.sh bash jupyter nbconvert --to html --execute ./scripts/e2e_test_notebook.ipynb > /dev/null
		grep -q -i "successfulrun" ./scripts/e2e_test_notebook.html
		_check_fail_test $? $LINENO
		rm -f ./scripts/e2e_test_notebook.html
	}

	it_starts_jupyter_server() {
		CONTAINER_NAME=e2e_plaster_test_it_starts_jupyter_server ERISYON_HEADLESS=1 JUP=1 DEV=1 IMAGE=plaster:e2e ./plaster.sh jupyter > __e2e.log 2>&1 &
		_sleep 8
		docker ps | grep -i "e2e_plaster_test_it_starts_jupyter_server" > /dev/null
		_check_fail_test $? $LINENO
		grep -q -i "http://127.0.0.1:8080/?token=" __e2e.log
		_check_fail_test $? $LINENO
		docker rm -f e2e_plaster_test_it_starts_jupyter_server
		rm __e2e.log
	}

	_run "it_passes_tests" || _fail_test $LINENO
	_run "it_gens_and_runs" || _fail_test $LINENO
	_run "it_starts_plaster_container_as_non_dev" || _fail_test $LINENO
	_run "it_starts_plaster_container_as_dev" || _fail_test $LINENO
	_run "it_runs_jupyter" || _fail_test $LINENO
	_run "it_starts_jupyter_server" || _fail_test $LINENO
}

_run "test_plaster_docker_container" || _fail_test $LINENO
