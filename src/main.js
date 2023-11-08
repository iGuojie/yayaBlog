import './assets/reset.css'

import { createApp } from 'vue';
import App from './App.vue'

// 导入markdown编译器相关插件
import VueMarkdownEditor from '@kangc/v-md-editor';
import vuepressTheme from '@kangc/v-md-editor/lib/theme/vuepress.js';
import VMdPreview from '@kangc/v-md-editor/lib/preview';
import '@kangc/v-md-editor/lib/style/base-editor.css';
import '@kangc/v-md-editor/lib/theme/style/vuepress.css';
import createKatexPlugin from '@kangc/v-md-editor/lib/plugins/katex/cdn';
import Prism from 'prismjs';
import hljs from 'highlight.js';
import VMdEditor from '@kangc/v-md-editor';
VueMarkdownEditor.use(vuepressTheme, {
    Prism
}, {
    config: {
        toc: {
            includeLevel: [1, 2],
        },
    },
}).use(createKatexPlugin());

VMdPreview.use(vuepressTheme, {
    Hljs: hljs,
}).use(createKatexPlugin());
const app = createApp(App);
app.use(VueMarkdownEditor);
app.use(VMdPreview);
app.mount('#app')
