# ipbloit

Operation Manual for “Stickchop”

Description
	Our “Stickchop” application is developed to help and guide people to hold chopsticks in a proper way. It uses hand detection to recognize correct hand posture and predict the appropriate way to hold the chopstick. Together with straight line detection, Stickchop can use both hand gesture and chopstick position to evaluate user chopstick holding style. Everyone will now be able to tell if they are using chopsticks right or wrong.

Preparation
1.	It is recommended for the user to position the camera in a way that the background is plain and simple.
2.	Lighting might affect the ability of the detection algorithm, so it is recommended to be in a properly lit setting.

Manual
1.	User should have an actual chopstick when using our application. (It might work on a long straight stick too.)

2.	Users should be showing the back of their hand to the camera, since the algorithm will be the most accurate when the user positions their hand in such a way.

3.	User hand should be at least filing half of the frame.

4.	Moving is not prohibited. If the user has the right posture, moving will still result in a good posture.

5.	If user has the right posture, the program will show the message “This is good posture +” and if user posture is wrong, the program will show the message “This is bad posture -”



*** Sometimes the user might already have the correct posture, but the algorithm didn’t show it that way. Slowly rotating hands might help the algorithm to recognize user posture.

*** We did test our algorithm on a few samples, but please acknowledge that it might not work or not as well for some individuals even if they got the right posture.

*** Our application decides good or bad posture based on a normal holding style. Some people might be comfortable with their own style, so don’t take the result too seriously.

Made by 

Mr.Patt Pootrakul

Mr.Pawaris Panyasombat

Mr.Thritsadi Chukiatsiri

Mr.Yuichi

Mr.Kota 
