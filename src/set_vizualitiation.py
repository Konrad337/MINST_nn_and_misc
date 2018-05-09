import itertools as it
from mnist_file_tools import get_bytes
from graphics import Point, Text, GraphWin, color_rgb, Rectangle


def print_next_image(columns, rows, file, win,
                     digits_per_side=5):

    width = win.getWidth()/300
    scale = win.getWidth()/300

    for k, l in it.product(range(digits_per_side), range(digits_per_side)):
        translation_x = k*width*columns
        translation_y = l*width*rows
        for j, i in it.product(range(columns), range(rows)):
            point = Rectangle(Point(i*scale - width/2 + translation_x,
                                    j*scale - width/2 + translation_y),
                              Point(i*scale + width/2 + translation_x,
                                    j*scale + width/2 + translation_y))
            color =  get_bytes(file, 1)
            point.setFill(color_rgb(color, color, color))
            point.draw(win)
# Prints next frame of digits, 25 be default


def print_next_label(file, win, digits_per_side=5):
    x = win.getWidth()/2
    y = 50
    for k, l in it.product(range(digits_per_side), range(digits_per_side)):
        translation_x = k*win.getWidth()/11 + x
        translation_y = l*win.getWidth()/11 + y
        label = get_bytes(file, 1)
        message = Text(Point(translation_x, translation_y), str(label))
        message.draw(win)
# Prints next set of labes


def clear(win):
    for item in win.items[:]:
        item.undraw()
    win.update()


def exit_button(win):
    exit_b = Rectangle(Point(win.getWidth()*5/12, win.getHeight()*9/12),
                       Point(win.getWidth()*7/12, win.getHeight()*11/12))
    exit_b.setFill("black")
    exit_b.draw(win)
    message = Text(Point(win.getWidth()/2, win.getHeight()*10/12), 'EXIT')
    message.setTextColor('white')
    message.setStyle('bold')
    message.setSize(30)
    message.draw(win)

    return exit_b


def vizualize(set, labels, how_many=10):

    train_set = open(set, 'rb')
    label_set = open(labels, 'rb')

    m_n = get_bytes(train_set)
    if m_n != 2051:
        raise Exception('Wrong magic number ' + str(m_n))
    m_n = get_bytes(label_set)
    if m_n != 2049:
        raise Exception('Wrong magic number ' + str(m_n))

    get_bytes(train_set)
    get_bytes(label_set)
    rows = get_bytes(train_set)
    columns = get_bytes(train_set)

    win = GraphWin('Data Set Number', 1200, 900, autoflush=False)

    for i in range(how_many):
        clear(win)
        print_next_image(columns, rows, train_set, win)
        print_next_label(label_set, win)
        exit_b = exit_button(win)
        mouse = win.getMouse()
        if exit_b.p1.x < mouse.x < exit_b.p2.x and exit_b.p1.y < mouse.y < exit_b.p2.y:
            break

    win.close()
    train_set.close()
