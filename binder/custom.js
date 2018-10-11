// Go to Running cell shortcut
//http://forums.fast.ai/t/jupyter-notebook-enhancements-tips-and-tricks/17064
Jupyter.keyboard_manager.command_shortcuts.add_shortcut('Alt-I', {
    help : 'Go to Running cell',
    help_index : 'zz',
    handler : function (event) {
        setTimeout(function() {
            // Find running cell and click the first one
            if ($('.running').length > 0) {
                //alert("found running cell");
                $('.running')[0].scrollIntoView();
            }}, 250);
        return false;
    }
});