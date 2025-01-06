# State space
The "FlappyBird-v0" environment, yields simple numerical information about the game's state as observations representing the game's screen.

## FlappyBird-v0
<li>
the last pipe's horizontal position
</li>
<li>
the last top pipe's vertical position
</li>
<li>
the last bottom pipe's vertical position
</li>
<li>
the next pipe's horizontal position
</li>
<li>
the next top pipe's vertical position
</li>
<li>
the next bottom pipe's vertical position
</li>
<li>
the next next pipe's horizontal position
</li>
<li>
the next next top pipe's vertical position
</li>
<li>
the next next bottom pipe's vertical position
</li>
<li>
player's vertical position
</li>
<li>
player's vertical velocity
</li>
<li>
player's rotation  
</li>

## Action space
<li>
0 - do nothing
</li>
<li>
1 - flap
</li>

## Rewards
<li>
+0.1 - every frame it stays alive
</li>
<li>
+1.0 - successfully passing a pipe
</li>
<li>
-1.0 - dying
</li>
<li>
−0.5 - touch the top of the screen
</li>


