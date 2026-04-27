Set objArgs = WScript.Arguments
targetPath = objArgs(0)
shortcutName = objArgs(1)
iconPath = objArgs(2)

Set objShell = CreateObject("WScript.Shell")
desktopPath = objShell.SpecialFolders("Desktop")

Set objShortcut = objShell.CreateShortcut(desktopPath & "\" & shortcutName)
objShortcut.TargetPath = targetPath
objShortcut.WorkingDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(targetPath)
objShortcut.WindowStyle = 1
objShortcut.IconLocation = iconPath & ",0"
objShortcut.Save
