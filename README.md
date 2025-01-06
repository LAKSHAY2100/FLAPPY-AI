## State space
The "FlappyBird-v0" environment, yields simple numerical information about the game's state as observations representing the game's screen.

# FlappyBird-v0
the last pipe's horizontal position
the last top pipe's vertical position
the last bottom pipe's vertical position
the next pipe's horizontal position
the next top pipe's vertical position
the next bottom pipe's vertical position
the next next pipe's horizontal position
the next next top pipe's vertical position
the next next bottom pipe's vertical position
player's vertical position
player's vertical velocity
player's rotation

# Action space
0 - do nothing
1 - flap

# Rewards
+0.1 - every frame it stays alive
+1.0 - successfully passing a pipe
-1.0 - dying
−0.5 - touch the top of the screen
