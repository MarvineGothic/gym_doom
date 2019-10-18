""" Helping utils """

""" Getting indices from buttons names"""


def buttons_to_actions(buttons, actions_set):
    actions = []

    for i in range(len(buttons)):
        if buttons[i] in actions_set:
            actions.append(actions_set.index(buttons[i]))
        else:
            raise ValueError('No such action corresponding to button ' + str(buttons[i]))
    return actions


""" Getting button names from action set """


def action_to_buttons(available_buttons, action_set):
    buttons_set = []
    for i in range(len(action_set)):
        if action_set[i]:
            buttons_set.append(available_buttons[i])
    return buttons_set
