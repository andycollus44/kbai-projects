#!/bin/bash

function _select_action() {
  PS3='Select the action: '
  options=('Test' 'Submit')

  select choice in "${options[@]}"
  do
    break
  done

  echo ${choice}
}

function _select_project() {
  PS3='Select the project: '
  options=('P1' 'P2' 'P3')

  select choice in "${options[@]}"
  do
    break
  done

  echo ${choice}
}

function _generate_assignment_command() {
  action=$1
  project=$2

  if [[ ${action} == 'Test' ]]; then
    case ${project} in
      'P1')
        cmd='error-check'
        ;;
      'P2')
        cmd='error-check-2'
        ;;
      'P3')
        cmd='error-check-3'
        ;;
    esac
  else
    cmd=${project}
  fi

  echo ${cmd}
}

action=$(_select_action)
echo

project=$(_select_project)
echo

echo "Executing action '${action}' for project '${project}'"
read -p "Continue? [y/n]: " continue

if [[ ! ${continue} =~ ^[Y/y]$ ]]; then echo; echo 'Bye!'; exit 1; fi

cmd=$(_generate_assignment_command ${action} ${project})

python submit.py --provider gt --assignment ${cmd} --files RavensProblemSolver.py RavensVisualProblem.py RavensTransformation.py RavensShape.py
