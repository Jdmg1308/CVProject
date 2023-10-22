
import pyautogui
import cv2


pointer_finger = int(8)
thumb = int(4)
index_y = 0
index_x = 0

screen_width, screen_height = pyautogui.size()

#land_mark must be a hand method usually hand.landmark
#width of the screen
#height of the screen
#current_img is current frame
def move_mouse(current_img, land_marks, width, height):
    for land_mark in enumerate(land_marks):
        x = int(land_mark.x * width)
        y = int(land_mark.y * height)

        if land_mark == pointer_finger:
            cv2.circle(img=current_img, center=(x,y), radius = 10, color=(100,100,100))
            index_x = screen_width/width * x
            index_y = screen_height/height * y
            pyautogui.moveTo(index_x, index_y)

        if land_mark == thumb:
            cv2.circle(img=current_img, center=(x,y), radius = 10, color=(100,100,100))
            thumb_y = screen_height/height * y
            
            ##calls mohammads method.
            if abs(thumb_y - index_y < 20):
                mouse_click(index_x, index_y)


#regesters click when gesture goes from point to close hand
def mouse_click(position_x, position_y):
    #click at x and y coordinates
    pyautogui.click(position_x, position_y)
    pyautogui.sleep(1)

###some methods to implement. just figure out gestures that make 
# sense or are differennt enough and map it to them

#selects all text on screen
def copy_all():
    pyautogui.hotkey("ctrlleft", "a")
    pyautogui.hotkey("ctrlleft", "c")
    
#pastes text (prob needs a text box)
def paste():
    pyautogui.hotkey("ctrlleft", "v")

#scrolls x pixels
def scroll(x):
    pyautogui.scroll(x)

# lets you reset mouse position without moving hand back in 1 sec
def reset_mouse():
    pyautogui.moveTo(0, 0, duration = 1)

# lets you write in textboxes. ***need to click on textbox for it to work***
def script():
    pyautogui.typewrite("write stuff. idk") #maybe have message asking for help? lol
    pyautogui.alert(text='wanna kms', title='HELP', button='L BOZO')  #creates popup

#Displays a message box with text input, and OK & Cancel buttons. Returns the text entered, or None if Cancel was clicked
#can use wth script function i guess
def create_textbox():
    pyautogui.prompt(text='', title='' , default='')

#
def bad_words():
    pyautogui.password(text='', title='', default='', mask='*')


#you trying too hard (what do these methods do (crying emoji))
#i wrote epic comments. just read em. that 2nd line for script is where its at
#ez mohammad (cool guy with glasses emoji)
#yessir. also the sleep method is in seconds so idk if you want a 2 sec pause after clicks oh fr maybe just 1 sec then
## make other methods if you want. the more the merrier since the gesture stuff should work later.

#if you want to help me Im gonna make a master file to add everything into it. FR it starting to work (great job on the methods)
