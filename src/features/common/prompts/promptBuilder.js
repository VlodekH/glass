const { profilePrompts } = require('./promptTemplates.js');

function buildSystemPrompt(promptParts, customPrompt = '', googleSearchEnabled = true, conversationHistory = '') {
    const sections = [promptParts.intro, '\n\n', promptParts.formatRequirements];

    if (googleSearchEnabled) {
        sections.push('\n\n', promptParts.searchUsage);
    }

    const outputInstructions = promptParts.outputInstructions.replace(
        '{{CONVERSATION_HISTORY}}',
        conversationHistory || 'No conversation history available.'
    );

    sections.push(
        '\n\n',
        promptParts.content,
        '\n\nUser-provided context\n-----\n',
        customPrompt || 'No custom context provided.',
        '\n-----\n\n',
        outputInstructions
    );

    return sections.join('');
}

function getSystemPrompt(profile, customPrompt = '', googleSearchEnabled = true, conversationHistory = '') {
    const promptParts = profilePrompts[profile] || profilePrompts.interview;
    return buildSystemPrompt(promptParts, customPrompt, googleSearchEnabled, conversationHistory);
}

module.exports = {
    getSystemPrompt,
};
