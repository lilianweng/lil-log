"use strict";

function _typeof(obj) { if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    define(['exports'], factory);
  } else if ((typeof exports === "undefined" ? "undefined" : _typeof(exports)) === 'object' && typeof exports.nodeName !== 'string') {
    factory(exports);
  } else {
    factory(root.commonJsStrict = {});
  }
})(typeof self !== 'undefined' ? self : void 0, function (exports) {
  exports.addBackToTop = addBackToTop; // FUNCTION START

  'use strict';

  function addBackToTop() {
    var params = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
    var _params$backgroundCol = params.backgroundColor,
        backgroundColor = _params$backgroundCol === void 0 ? '#000' : _params$backgroundCol,
        _params$cornerOffset = params.cornerOffset,
        cornerOffset = _params$cornerOffset === void 0 ? 20 : _params$cornerOffset,
        _params$diameter = params.diameter,
        diameter = _params$diameter === void 0 ? 56 : _params$diameter,
        _params$ease = params.ease,
        ease = _params$ease === void 0 ? inOutSine : _params$ease,
        _params$id = params.id,
        id = _params$id === void 0 ? 'back-to-top' : _params$id,
        _params$innerHTML = params.innerHTML,
        innerHTML = _params$innerHTML === void 0 ? '<svg viewBox="0 0 24 24"><path d="M7.41 15.41L12 10.83l4.59 4.58L18 14l-6-6-6 6z"></path></svg>' : _params$innerHTML,
        _params$onClickScroll = params.onClickScrollTo,
        onClickScrollTo = _params$onClickScroll === void 0 ? 0 : _params$onClickScroll,
        _params$scrollContain = params.scrollContainer,
        scrollContainer = _params$scrollContain === void 0 ? document.body : _params$scrollContain,
        _params$scrollDuratio = params.scrollDuration,
        scrollDuration = _params$scrollDuratio === void 0 ? 100 : _params$scrollDuratio,
        _params$showWhenScrol = params.showWhenScrollTopIs,
        showWhenScrollTopIs = _params$showWhenScrol === void 0 ? 1 : _params$showWhenScrol,
        _params$size = params.size,
        size = _params$size === void 0 ? diameter : _params$size,
        _params$textColor = params.textColor,
        textColor = _params$textColor === void 0 ? '#fff' : _params$textColor,
        _params$zIndex = params.zIndex,
        zIndex = _params$zIndex === void 0 ? 1 : _params$zIndex;
    var scrollContainerIsBody = scrollContainer === document.body;
    var scrollDocumentElement = scrollContainerIsBody && document.documentElement;
    appendStyles();
    var upEl = appendElement();
    var hidden = true;
    var scrollEmitter = scrollContainerIsBody ? window : scrollContainer;
    scrollEmitter.addEventListener('scroll', adapt);
    adapt();

    function adapt() {
      getScrollTop() >= showWhenScrollTopIs ? show() : hide();
    }

    function show() {
      if (!hidden) {
        return;
      }

      upEl.className = '';
      hidden = false;
    }

    function hide() {
      if (hidden) {
        return;
      }

      upEl.className = 'hidden';
      hidden = true;
    }

    function appendElement() {
      var upEl = document.createElement('div');
      upEl.id = id;
      upEl.className = 'hidden';
      upEl.innerHTML = innerHTML;
      upEl.addEventListener('click', function (event) {
        event.preventDefault();
        scrollUp();
      });
      document.body.appendChild(upEl);
      return upEl;
    }

    function appendStyles() {
      var svgSize = Math.round(0.43 * size);
      var svgTop = Math.round(0.29 * size);
      var styles = '#' + id + '{background:' + backgroundColor + ';-webkit-border-radius:50%;-moz-border-radius:50%;border-radius:50%;bottom:' + cornerOffset + 'px;-webkit-box-shadow:0 2px 5px 0 rgba(0,0,0,.26);-moz-box-shadow:0 2px 5px 0 rgba(0,0,0,.26);box-shadow:0 2px 5px 0 rgba(0,0,0,.26);color:' + textColor + ';cursor:pointer;display:block;height:' + size + 'px;opacity:1;outline:0;position:fixed;right:' + cornerOffset + 'px;-webkit-tap-highlight-color:transparent;-webkit-touch-callout:none;-webkit-transition:bottom .2s,opacity .2s;-o-transition:bottom .2s,opacity .2s;-moz-transition:bottom .2s,opacity .2s;transition:bottom .2s,opacity .2s;-webkit-user-select:none;-moz-user-select:none;-ms-user-select:none;user-select:none;width:' + size + 'px;z-index:' + zIndex + '}#' + id + ' svg{display:block;fill:currentColor;height:' + svgSize + 'px;margin:' + svgTop + 'px auto 0;width:' + svgSize + 'px}#' + id + '.hidden{bottom:-' + size + 'px;opacity:0}';
      var styleEl = document.createElement('style');
      styleEl.appendChild(document.createTextNode(styles));
      document.head.insertAdjacentElement('afterbegin', styleEl);
    }

    function scrollUp() {
      var scrollTo = typeof onClickScrollTo === 'function' ? onClickScrollTo() : onClickScrollTo;
      var _window = window,
          performance = _window.performance,
          requestAnimationFrame = _window.requestAnimationFrame;

      if (scrollDuration <= 0 || typeof performance === 'undefined' || typeof requestAnimationFrame === 'undefined') {
        return setScrollTop(scrollTo);
      }

      var start = performance.now();
      var initScrollTop = getScrollTop();
      var pxsToScrollBy = initScrollTop - scrollTo;
      requestAnimationFrame(step);

      function step(timestamp) {
        var progress = Math.min((timestamp - start) / scrollDuration, 1);
        setScrollTop(initScrollTop - Math.round(ease(progress) * pxsToScrollBy));

        if (progress < 1) {
          requestAnimationFrame(step);
        }
      }
    }

    function getScrollTop() {
      return scrollContainer.scrollTop || scrollDocumentElement && document.documentElement.scrollTop || 0;
    }

    function setScrollTop(value) {
      scrollContainer.scrollTop = value;

      if (scrollDocumentElement) {
        document.documentElement.scrollTop = value;
      }
    }

    function inOutSine(n) {
      return 0.5 * (1 - Math.cos(Math.PI * n));
    }
  } // FUNCTION END

});